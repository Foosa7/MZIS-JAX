"""Ablation: does thermal crosstalk still vanish under a physically correct kernel?

The earlier result -- crosstalk contributes ~0.2% beyond a bias-fitted diagonal
model -- was obtained with an isotropic `exp(-d/lambda)` kernel on *grid index*
distance. The tapeout geometry says that kernel could not have fit: the heater
lattice is anisotropic (254 um within a Clements column vs 1125 um between
columns, both "distance 1" on the grid), the 190 um heaters are line sources at
254 um separation rather than far-field points, and a mesh spanning millimetres
carries a global gradient no pairwise kernel can express.

This script re-runs the ablation with the physical kernel from `src.thermal`
against the same measured datasets, on a held-out split, and reports whether the
null survives.

Usage:

    python scripts/fit_crosstalk.py \
        --dataset /path/to/dataset_20260324_8mode_0broken_1600steps.json \
        --resistance /path/to/8-mode_20260209_resistance_params.json

Note the datasets are a colleague's measurements; this is a model-comparison run
against them, not a claim on them.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.engine import Engine  # sets x64 + preallocation flags; import first

import jax
import jax.numpy as jnp
import optax

from src.geometry import MeshGeometry
from src.thermal import ThermalKernel

TWO_PI = 2.0 * np.pi


# ──────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────

def load_dataset(path, geom):
    with open(path) as fh:
        raw = json.load(fh)
    meta = raw["metadata"]
    if meta["mzi_labels"] != geom.mzi_ids:
        raise SystemExit(
            "dataset MZI order does not match engine order:\n  %s\n  %s"
            % (meta["mzi_labels"], geom.mzi_ids)
        )
    if "simulator" in meta or "mismatch_profile" in meta:
        print("!! metadata carries a simulator block -- this dataset is SIMULATED,")
        print("!! so a null crosstalk result is guaranteed and meaningless.")

    samples = raw["samples"]
    currents = np.array([s["currents_mA"] for s in samples], dtype=np.float64)
    ports = np.array([s["input_port"] for s in samples], dtype=np.int32) - 1
    outputs = np.array([s["outputs_mW"] for s in samples], dtype=np.float64)
    # Detector dark offset puts a few hundred entries slightly below zero; clip
    # before normalising so the Bhattacharyya fidelity stays real.
    n_neg = int((outputs < 0).sum())
    outputs = np.clip(outputs, 0.0, None)
    outputs = outputs / np.clip(outputs.sum(axis=1, keepdims=True), 1e-12, None)
    return meta, currents, ports, outputs, n_neg


def load_phase_calibration(path, geom, phase_scale):
    """Per-heater omega (rad/mW) and static fringe phase from node isolation.

    These are not trustworthy as absolute values -- node isolation measures each
    heater's fringe against whatever cross/bar state the *other* heaters happen
    to be in, so the offset is conflated and a global fit is what actually
    recovers the static bias. But as a *starting point* they are worth a great
    deal: the bias landscape wraps at 2*pi and is thick with local minima, and
    starting every heater at zero drops the optimiser into an arbitrary basin.

    Returns `(gain0, bias0, known)` where gain0 is omega/phase_scale.
    """
    gain0 = np.ones(geom.n_heaters)
    bias0 = np.zeros(geom.n_heaters)
    known = np.zeros(geom.n_heaters, dtype=bool)
    if not os.path.exists(path):
        return gain0, bias0, known
    with open(path) as fh:
        cal = json.load(fh).get("phase_calibration", {})
    for i, name in enumerate(geom.heater_names):
        if name not in cal:
            continue
        pp = cal[name].get("phase_params", {})
        if "omega" not in pp:
            continue
        gain0[i] = float(pp["omega"]) / phase_scale
        bias0[i] = float(pp.get("phase", 0.0))
        known[i] = True
    return gain0, bias0, known


def load_resistance(path, geom):
    """c_res / alpha_res per heater, in the interleaved heater ordering."""
    with open(path) as fh:
        params = json.load(fh)
    c = np.ones(geom.n_heaters)
    a = np.zeros(geom.n_heaters)
    for i, name in enumerate(geom.heater_names):
        if name in params:
            c[i] = float(params[name].get("c_res", 1.0))
            a[i] = float(params[name].get("alpha_res", 0.0))
    return c, a


# ──────────────────────────────────────────────────────────────────────────
# Forward model
# ──────────────────────────────────────────────────────────────────────────

def make_forward(engine, kernel, c_res, alpha_res, phase_scale, active, mode,
                 mean_total_power=1.0):
    """currents (mA) + input port -> predicted normalised output distribution."""

    c_res = jnp.asarray(c_res)
    alpha_res = jnp.asarray(alpha_res)
    active = jnp.asarray(active)

    def powers_of(currents):
        i_sq = (currents * active) ** 2
        return i_sq * c_res * (1.0 + alpha_res * i_sq)

    def phases_of(params, currents):
        p = powers_of(currents)
        base = p * phase_scale * jnp.exp(params["log_gain"]) * active
        ph = base + params["bias"]
        if mode != "none":
            c = kernel.coupling(params["kernel"], mode)
            ph = ph + p @ c
        if "global_beta" in params:
            # Scaled by the dataset's mean total dissipated power so the
            # coefficient is O(1 rad) and starts on the same footing as `bias`.
            ph = ph + params["global_beta"] * (p.sum() / mean_total_power)
        return ph

    def single(params, currents, port):
        ph = phases_of(params, currents)
        u = engine.compute_full_unitary(ph[0::2], ph[1::2])
        probs = jnp.abs(u[:, port]) ** 2
        probs = probs * jnp.exp(params["log_eff"])
        return probs / jnp.clip(probs.sum(), 1e-12)

    return jax.vmap(single, in_axes=(None, 0, 0)), phases_of


def init_params(geom, kernel, mode, fit_global, seed=0, bias_spread=0.0, cal=None):
    """Diagonal params, started from node isolation where a variant says to.

    The per-heater zero-current phase bias is the dominant term and its
    landscape wraps at 2*pi, so a single start from zero lands in whichever
    basin happens to be nearest. Restarts explore: the calibration phase as
    measured, its negation (the fringe fit does not fix the sign convention),
    each of those with pi added on the theta heaters (the engine's mesh identity
    is theta = pi, not 0), and finally uniform random.
    """
    rng = np.random.default_rng(seed)
    gain = np.zeros(geom.n_heaters)
    bias = np.zeros(geom.n_heaters)

    if cal is not None and bias_spread <= 0:
        gain0, bias0, known = cal
        gain = np.log(np.where(known, gain0, 1.0))
        sign = -1.0 if (seed // 2) % 2 else 1.0
        bias = sign * np.where(known, bias0, 0.0)
        if seed % 2:
            bias = bias + np.pi * geom.heater_is_theta
    elif bias_spread > 0:
        bias = rng.uniform(-bias_spread, bias_spread, geom.n_heaters)

    params = {
        "log_gain": jnp.asarray(gain),
        "bias": jnp.asarray(bias),
        "log_eff": jnp.zeros(geom.n_modes),
    }
    if mode != "none":
        params["kernel"] = kernel.init_params(mode)
    if fit_global:
        params["global_beta"] = jnp.zeros(())
    return params


def fidelity(pred, target):
    """Classical (Bhattacharyya) fidelity, averaged over samples."""
    return jnp.mean(jnp.sum(jnp.sqrt(jnp.clip(pred, 0.0, None) * target), axis=1) ** 2)


# ──────────────────────────────────────────────────────────────────────────

def run(name, mode, fit_global, engine, kernel, geom, data, args,
        params=None, epochs=None, quiet=False):
    currents, ports, outputs, split = data
    c_res, alpha_res, phase_scale, active, mean_total_power = args["cal"]
    epochs = epochs if epochs is not None else args["epochs"]

    forward, phases_of = make_forward(
        engine, kernel, c_res, alpha_res, phase_scale, active, mode, mean_total_power
    )
    if params is None:
        params = init_params(geom, kernel, mode, fit_global)

    tr, va = split
    ctr, ptr, otr = currents[tr], ports[tr], outputs[tr]
    cva, pva, ova = currents[va], ports[va], outputs[va]

    def loss_fn(p, c, prt, o):
        return jnp.mean((forward(p, c, prt) - o) ** 2)

    n_steps = max(1, epochs * max(1, ctr.shape[0] // args["batch"]))
    schedule = optax.cosine_decay_schedule(args["lr"], n_steps, alpha=0.02)
    opt = optax.adam(schedule)
    state = opt.init(params)

    @jax.jit
    def step(params, state, c, prt, o):
        loss, grads = jax.value_and_grad(loss_fn)(params, c, prt, o)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    @jax.jit
    def evaluate(params):
        return fidelity(forward(params, cva, pva), ova), loss_fn(params, cva, pva, ova)

    f0, _ = evaluate(params)
    n = ctr.shape[0]
    bs = min(args["batch"], n)
    rng = np.random.default_rng(0)
    t0 = time.time()
    best = (float(f0), params)

    for epoch in range(epochs):
        perm = rng.permutation(n)
        for k in range(0, n - bs + 1, bs):
            idx = perm[k:k + bs]
            params, state, _ = step(params, state, ctr[idx], ptr[idx], otr[idx])
        fid, mse = evaluate(params)
        if float(fid) > best[0]:
            best = (float(fid), params)
        if not quiet and (epoch % max(1, epochs // 8) == 0 or epoch == epochs - 1):
            print(
                "    epoch %5d  val fidelity %.5f  val MSE %.3e  (%.0fs)"
                % (epoch, float(fid), float(mse), time.time() - t0)
            )

    fid, mse = best[0], float(evaluate(best[1])[1])
    if not quiet:
        print("  %-34s val fidelity %.5f   val MSE %.4e" % (name, fid, mse))
    return fid, best[1], params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--resistance", required=True)
    ap.add_argument("--calibration", default="node-isolation/8-mode-autocal-20260209.json")
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--quad", type=int, default=8)
    ap.add_argument("--cache-dir", default=".fitcache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--ideal-couplers", action="store_true",
                    help="skip node-isolation beamsplitter errors (e_l = e_r = 0)")
    args = ap.parse_args()

    geom = MeshGeometry.load(args.modes)
    meta, currents, ports, outputs, n_neg = load_dataset(args.dataset, geom)
    c_res, alpha_res = load_resistance(args.resistance, geom)
    phase_scale = TWO_PI / float(meta["p_2pi_mW"])

    # A-column phi heaters are never driven; keep them as crosstalk victims but
    # not as actuators or sources.
    active = (currents != 0.0).any(axis=0).astype(np.float64)
    print("dataset      : %s" % os.path.basename(args.dataset))
    print("samples      : %d   modes: %d   active heaters: %d/%d"
          % (len(currents), args.modes, int(active.sum()), geom.n_heaters))
    print("p_2pi        : %.5f mW  ->  phase_scale %.6f rad/mW"
          % (meta["p_2pi_mW"], phase_scale))
    print("dark offset  : %d negative power readings clipped to 0" % n_neg)

    engine = Engine(n_modes=args.modes)
    if not args.ideal_couplers and os.path.exists(args.calibration):
        engine.load_calibration_errors(args.calibration)
        print("couplers     : node-isolation e_l=e_r, mean %.4f"
              % float(np.mean(np.asarray(engine.e_l))))
    else:
        print("couplers     : ideal")

    kernel = ThermalKernel(geom, n_quad=args.quad)
    print("geometry     : %s" % json.dumps(geom.summary()))

    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(currents))
    n_val = int(round(args.val_frac * len(currents)))
    split = (perm[n_val:], perm[:n_val])
    data = (currents, ports, outputs, split)
    i_sq = (currents * active) ** 2
    mean_total_power = float((i_sq * c_res * (1.0 + alpha_res * i_sq)).sum(axis=1).mean())
    print("mean total power in mesh: %.2f mW" % mean_total_power)

    args_d = {
        "cal": (c_res, alpha_res, phase_scale, active, mean_total_power),
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
    }

    cal = load_phase_calibration(args.calibration, geom, phase_scale)
    print("phase cal    : %d/%d heaters have omega + fringe phase"
          % (int(cal[2].sum()), geom.n_heaters))

    # Stage 1 is the expensive part and does not depend on the kernel, so cache
    # it keyed on everything that would change the answer.
    cache_key = "%s_%s_%d_%d_%g_%s" % (
        os.path.basename(args.dataset), args.modes, args.epochs, args.restarts,
        args.lr, "ideal" if args.ideal_couplers else "cal",
    )
    cache_path = os.path.join(args.cache_dir, "diag_%s.npz" % cache_key)

    if os.path.exists(cache_path) and not args.no_cache:
        blob = np.load(cache_path)
        diag_params = {
            "log_gain": jnp.asarray(blob["log_gain"]),
            "bias": jnp.asarray(blob["bias"]),
            "log_eff": jnp.asarray(blob["log_eff"]),
        }
        base = float(blob["fidelity"])
        print("\n=== stage 1: loaded cached diagonal fit (%.5f) ===" % base)
        best_diag = (base, diag_params)
        args.restarts = 0

    print("\n=== stage 1: diagonal model, %d restarts ===" % args.restarts)
    best_diag = best_diag if args.restarts == 0 else (-1.0, None)
    for r in range(args.restarts):
        # First four restarts walk the calibration sign / theta-offset variants;
        # anything beyond that is random.
        p0 = init_params(geom, kernel, "none", False, seed=r,
                         bias_spread=0.0 if r < 4 else np.pi,
                         cal=cal)
        fid, params, _ = run("restart %d" % r, "none", False, engine, kernel, geom,
                             data, args_d, params=p0, quiet=True)
        tag = ["cal", "cal+pi", "-cal", "-cal+pi"][r] if r < 4 else "random"
        print("  restart %d (%-8s) val fidelity %.5f%s"
              % (r, tag, fid, "  *" if fid > best_diag[0] else ""))
        if fid > best_diag[0]:
            best_diag = (fid, params)

    base, diag_params = best_diag
    print("\n  diagonal model (best):  val fidelity %.5f" % base)

    if args.restarts:
        os.makedirs(args.cache_dir, exist_ok=True)
        np.savez(cache_path, fidelity=base,
                 **{k: np.asarray(v) for k, v in diag_params.items()})
        print("  cached to %s" % cache_path)

    # Every kernel run starts from the same converged diagonal optimum, so the
    # comparison measures what the kernel adds rather than optimiser luck.
    print("\n=== stage 2: kernels warm-started from the diagonal optimum ===")
    results = {"diagonal only": base}
    for name, mode, fit_global in (
        ("+ grid isotropic kernel", "grid_isotropic", False),
        ("+ physical kernel", "image_sink", False),
        ("+ physical kernel + global", "image_sink", True),
    ):
        print("\n  [%s]" % name)
        p0 = dict(diag_params)
        # Start the kernel at (numerically) zero amplitude so the warm start
        # reproduces the diagonal optimum exactly. Otherwise the run begins
        # *below* the baseline and a null result is ambiguous between "the
        # kernel adds nothing" and "the optimiser never recovered". Shape
        # parameters still start at their physical values.
        p0["kernel"] = dict(kernel.init_params(mode))
        p0["kernel"]["amp"] = jnp.zeros(())
        if fit_global:
            p0["global_beta"] = jnp.zeros(())
        fid, params, final = run(name, mode, fit_global, engine, kernel, geom, data,
                                 args_d, params=p0)
        results[name] = fid
        print("    kernel: %s" % json.dumps(kernel.report(params["kernel"], mode)))
        # `params` is the best-fidelity epoch, which for a true null is the
        # warm start itself. Print the final-epoch params too, so "the optimiser
        # explored and came back" is distinguishable from "nothing ever moved".
        for tag, kp in (("best", params["kernel"]), ("final", final["kernel"])):
            if mode == "image_sink":
                print("    %-5s amp = %+.5f (peak %.3g rad/mW), h = %.1f um, a = %.1f um"
                      % (tag, float(kp["amp"]),
                         float(kp["amp"]) * kernel.AMP_UNIT,
                         float(jnp.exp(kp["log_h"])), float(jnp.exp(kp["log_a"]))))
            else:
                print("    %-5s amp = %+.5f (peak %.3g rad/mW), lam = %.3f grid"
                      % (tag, float(kp["amp"]),
                         float(kp["amp"]) * kernel.AMP_UNIT,
                         float(jnp.exp(kp["log_lam"]))))
        if fit_global:
            print("    global_beta: best %+.5g  final %+.5g rad per mean-total-power"
                  % (float(params["global_beta"]), float(final["global_beta"])))

    print("\n=== marginal contribution over the diagonal model ===")
    for name, fid in results.items():
        if name == "diagonal only":
            continue
        print("  %-30s %+.4f  (%.5f -> %.5f)" % (name, fid - base, base, fid))
    print("\nThe earlier isotropic-kernel result was +0.002.")


if __name__ == "__main__":
    main()
