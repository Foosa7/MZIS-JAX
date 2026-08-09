# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A JAX-based simulator + Tkinter visualizer for Clements-mesh Mach-Zehnder interferometers (MZI), used as a digital twin of a physical 8-mode photonic chip. It simulates classical power flow, coherent field propagation, and multi-photon Fock propagation (permanents), and can decompose arbitrary target unitaries into per-MZI phase settings.

## Commands

```bash
pip install -r requirements.txt
python main.py                         # launch the GUI (needs a display; Tk)

python tests/test_integrations.py      # bs_error, phase constraints, nullification
python tests/test_calibration_mapping.py  # phase -> electrical power -> DAC current
python scripts/test_opt.py             # single-run Adam phase optimization
python scripts/test_coherent_opt.py    # coherent routing via StateRouter
python scripts/test_quantum_opt.py     # Fock-marginal routing via StateRouter
python scripts/test_hom_diagnostic.py  # HOM dip sweep

python scripts/generate_unitaries.py   # writes unitary/random_unitary_{8,12,16}x*.npy
python scripts/generate_switching.py   # writes unitary/random_switching_*.npy

# physical layout -> layout/<n>mode_geometry.json  (needs `pip install gdstk`)
python scripts/extract_layout_geometry.py --gds ~/Downloads/8x8-chip.gds --modes 8
python scripts/extract_layout_geometry.py --gds ~/layout/12x12_desgin_final.gds --modes 12
python -m src.geometry                 # print the extracted lattice summary

# thermal-crosstalk ablation against measured data
python scripts/fit_crosstalk.py --dataset <dataset.json> --resistance <resistance_params.json>
```

There is no pytest, no test runner, and no lint config. Files under `tests/` and `scripts/` are print-based `__main__` scripts — "running a single test" means running the file, or calling one of its `test_*()` functions. Run them from the repo root (`tests/test_hom.py` has no `sys.path` bootstrap; the others insert the repo root themselves).

`requirements.txt` is incomplete: `src/routing.py` also needs `optax` and `scipy`.

`tests/test_hom.py` is stale — it uses an `Engine.phases` attribute and a 1-arg `propagate_fock` that no longer exist. Fix or ignore it; don't take it as a description of the current API.

## Architecture

Four layers, deliberately separated by purity:

- `src/engine.py` — **stateless, pure JAX**. `Engine` holds only topology (layout, mode indices, beamsplitter errors); all phase values are passed in as arrays. Everything hot is `@jit` + `lax.scan`.
- `src/routing.py` — `StateRouter`: gradient-based inverse design on top of `Engine`, using `optax.adam` under `jax.vmap` for hundreds of parallel random restarts, then dedup + sort by loss.
- `decompose/pnn.py` — **NumPy/sympy**, not JAX. Analytic Clements decomposition/reconstruction. Deliberately outside the differentiable path.
- `src/gui.py` — Tkinter/matplotlib. **Owns the mutable phase state** (`self.phases[mzi_id] = {'theta','phi'}`) and orchestrates everything else.

`src/engine.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `jax_enable_x64=True` at import time. Everything is complex128/float64; import `src.engine` before other JAX work so the x64 flag is set first.

### Mesh layout and array ordering

`Engine._define_layout()` builds `n_modes` columns in a checkerboard: even columns pair modes (0,1),(2,3),…; odd columns pair (1,2),(3,4),…. Columns are named `A, B, C, …` and MZIs within a column `1, 2, …`, giving ids like `G4`. Those ids match the heater names in the calibration JSON (`G4_theta`).

`engine.mzi_ids` defines the canonical flat ordering (column-major). Every `thetas`/`phis`/`e_l`/`e_r` array is indexed in that order, `_col_slices[i]` gives `(start, count)` into it, and `gui._get_phase_arrays()` is the single bridge from the GUI dict to that ordering. If you touch the layout, all four arrays and the slice table move together.

### MZI transfer-matrix convention (fragile)

Each MZI is `BS(e_r) · diag(e^{iθ}, 1) · BS(e_l)` with `R = √(1+ε)/√2`, `T = i√(1-ε)/√2`, followed by two convention fixups: negate the second column, then multiply the first column by `e^{iφ}` (external phase on the top input). Those fixups exist so the error model reduces exactly to the original ideal MZI at ε=0 — removing them silently breaks the HOM and boson-sampling demos. See `reports/neurophox_integration_report.md` and `reports/beamsplitter_math_derivation.md`.

**This inner block is duplicated verbatim** in `Engine._build_layer_matrix` (per-column, used by classical flow) and in the nested `mzi_body` of `Engine._compute_full_unitary` (whole mesh). Any change to the MZI physics must be applied to both or the two paths disagree.

`compute_full_unitary` pads `thetas`/`phis`/`e_l`/`e_r` by `max_mzis` before calling the jitted static method, because the fixed-shape `lax.scan` over columns dynamic-slices a full `max_mzis` window and masks the overrun. Call the public method, not `_compute_full_unitary` directly.

### Decomposition ↔ engine mapping

`decompose_clements(U, block='mzi')` returns `(phis, thetas, alphas)` indexed `[mode_top, column_pair]`. `gui._apply_unitary_decomposition` maps them onto the engine with:

- `p = col_idx // 2`, `q = mzi['mode_top']`
- `engine_theta = 2 * pnn_theta` (pnn's θ is half-angle), `engine_phi = pnn_phi`, both mod 2π
- `alphas` (the output phase screen) is currently **dropped** — reconstructions match power, not global output phases.

Mesh identity is `theta = π, phi = 0` for every MZI (see `_demo_clear`), not zeros.

### Quantum path

`propagate_fock` builds the full unitary, enumerates the output Fock basis with `combinations_with_replacement`, and evaluates `|Perm(U_sub)|²/normalizations` via `ryser_permanent` under `jax.lax.map` (sequential, to bound memory). `ryser_permanent` materializes all `2^n - 1` subsets, so cost is exponential in photon number — it is fine for the 2–3 photons the GUI uses and will explode beyond that. `StateRouter.optimize_quantum_routing_vmap` puts this whole thing inside an Adam loss, hence its much lower restart/iteration defaults.

### Hardware error model

`bs_error` accepts a scalar, an array (used for both sides), or an `(e_l, e_r)` tuple. `Engine.load_calibration_errors()` reads `node-isolation/8-mode-autocal-20260209.json` and derives per-MZI ε from the cosine-fit fringe visibility: `V = amplitude/offset`, `ε = √((C−A)/(C+A))`, keyed by `<mzi_id>_theta`. MZIs absent from the JSON (e.g. running a 12- or 16-mode mesh against 8-mode calibration) fall back to `default_e`. The GUI exposes this as the "Hardware Imperfections" dropdown.

The same JSON also carries `resistance_calibration` and `phase_params.omega`; `tests/test_calibration_mapping.py` shows the full chain phase → required electrical power (`θ/ω`) → DAC current (real positive root of `d·x³ + a·x² + c·x − P` with `x = I²`).

`routing.py` prints its backend and auto-scales restarts/iterations for GPU vs CPU at import time.

**Caution on those ε values.** They are derived from the cosine *fit*'s residual, not the raw fringe. Median 0.050, max 0.127, and they disagree with the raw min/max of the same sweep (median 0.038; `E2_theta` gives 0.095 from the fit and 0.000 from the data, whose fringe minimum is actually negative). A nonzero fringe minimum can come from loss, detector dark offset, or spurious light as readily as from coupler imbalance, and the chip's beamsplitters are 2×2 MMIs, whose splitting is geometrically robust. Treat `load_calibration_errors` as an upper bound, not a measurement; fits that *learn* ε land near 0.01. Pass `--ideal-couplers` to `fit_crosstalk.py` to check any result for sensitivity to this.

### Physical layout

`layout/<n>mode_geometry.json` holds real die coordinates, generated from the tapeout GDS by `scripts/extract_layout_geometry.py` and read by `src/geometry.py`. The two dies are different designs — the 8-mode is Nazca/SOI250 propagating along **x**, the 12-mode is gdsfactory/AMF propagating along **y** — so the extractor detects the propagation axis and emits a canonical `(propagation, mode)` frame.

|  | 8-mode | 12-mode |
|---|---|---|
| mode pitch | 127 µm | 127 µm |
| same-column neighbour | 254 µm | 254 µm |
| φ→θ within an MZI | 562.5 µm | 633.9 µm |
| Clements column pitch | 1125 µm | 1267.9 µm |
| heater length | 190 µm | 300 µm |
| arm segment between MMIs | 497.5 µm | 558.9 µm |

Each MZI is laid out `φ-heater | MMI | θ-heater | MMI` along propagation, matching the engine's `BS(e_r)·diag(e^{iθ},1)·BS(e_l)` with φ external. **Both heaters sit on the `mode_top` arm**, and mode 0 is at the high end of the mode axis on both dies. The extractor does not assume this — it predicts an (x, y) for every heater from `Engine`'s id convention and fails loudly if any predicted site is empty, so a passing run is a validation of the id↔coordinate join.

`MeshGeometry.heater_xy` is interleaved `[theta, phi]` per MZI in `Engine.mzi_ids` order. That is exactly the 56-vector ordering of `currents_mA` in the DigitalTwin datasets, and `mzi_ids` matches their `mzi_labels` element for element.

### Thermal crosstalk

`src/thermal.py` provides two kernels mapping heater power (mW) to phase (rad). `grid_isotropic` is the historical `amp·exp(-d/λ)` on *grid index* distance; `image_sink` is the physical one — a finite line source over an isothermal substrate at fitted depth `h`, line-integrated by quadrature over both the source heater and the victim arm segment, and evaluated **differentially** across the 127 µm arm pair.

The grid kernel cannot work on this device: 254 µm and 1125 µm neighbours are both "distance 1" in grid coordinates, a 4.4:1 anisotropy collapsed into one λ. Nor is the far-field approximation valid — the heaters are 190 µm long and 254 µm apart. `total_power_term` adds the rank-one direction no pairwise kernel can express: a mesh spanning millimetres has different heat paths to the TEC end to end.

### Fitting against measured data

`scripts/fit_crosstalk.py` fits per-heater gain + static bias + per-output efficiency through the engine forward, then warm-starts each kernel from that shared diagonal optimum so the ablation measures the kernel rather than optimiser luck.

**Initialisation is the whole ballgame.** The static-bias landscape wraps at 2π and is dense with local minima. Starting every bias at zero plateaus around **0.72** held-out fidelity and cannot even overfit 32 samples; seeding bias from `phase_calibration[<heater>].phase_params.phase` and gain from that heater's own `omega` reaches **0.995** on the same data and optimiser. If a fit of this mesh is stuck in the 0.7–0.85 range, suspect the basin before adding capacity.

Two data quirks: the measured `outputs_mW` contain a few hundred slightly negative readings (detector dark offset) that NaN any `sqrt`-based fidelity unless clipped, and column-A φ heaters are never driven — keep them as crosstalk victims with a free bias, but not as actuators or sources.

## Repo conventions

`.gitignore` excludes `*.npy` and `reports/`, but the existing files under `unitary/` and `reports/` are tracked. New matrices or reports need `git add -f` to be committed.
