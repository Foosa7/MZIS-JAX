# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A JAX-based photonic simulation engine and interactive Tkinter visualizer for Mach-Zehnder Interferometer (MZI) meshes in a Clements (checkerboard) layout. It simulates classical power flow, coherent field propagation, and quantum Fock-state propagation (Ryser permanents), and decomposes arbitrary target unitaries into physical MZI phase settings.

## Commands

```bash
pip install -r requirements.txt   # note: incomplete — routing.py also needs optax and scipy
python main.py                    # launch the interactive GUI

# Test / demo scripts (all are standalone print-based scripts, not pytest)
python tests/test_integrations.py        # bs_error, phase constraints, nullification
python tests/test_calibration_mapping.py # end-to-end phase -> DAC current mapping
python -m tests.test_hom                 # NOTE: stale, uses removed Engine.phases API

python scripts/test_opt.py               # single-restart gradient descent sanity check
python scripts/test_coherent_opt.py
python scripts/test_quantum_opt.py
python scripts/test_hom_diagnostic.py
python scripts/generate_unitaries.py     # regenerates unitary/random_unitary_{8,12,16}.npy
python scripts/generate_switching.py     # regenerates unitary/random_switching_*.npy
```

There is no test runner, linter, or build step. `tests/` and `scripts/` are indistinguishable in kind — both hold ad-hoc verification scripts. Scripts under `scripts/` and `tests/` insert the repo root into `sys.path` themselves; run them from the repo root.

## Architecture

**`src/engine.py` — pure functional simulation core.** `Engine` holds only *structure* (layout, MZI ids, beamsplitter errors), never phase state. Phases are always passed in as flat JAX arrays `(thetas, phis)` of length `n_mzis`, ordered column-by-column following `engine.mzi_ids`. This is what makes the whole pipeline JIT-able and differentiable — do not add mutable phase state to `Engine`.

- `_define_layout()` builds `n_modes` columns in a checkerboard pattern; MZI ids are `<ColumnLetter><k>` (`A1`, `B1`, `G4`, …) — these ids are the join key with hardware calibration JSON (`G4_theta`).
- `_build_layer_matrix` (one column) and `_compute_full_unitary` (whole mesh) contain **duplicated MZI transfer-matrix math**. Any change to the MZI physics must be applied to both. `_compute_full_unitary` pads columns to a uniform width and masks the padding with identity so `lax.scan` sees fixed shapes; `compute_full_unitary()` does that padding on the host side.
- MZI model: `U = BS(e_r) · diag(e^{iθ}, 1) · BS(e_l)`, then column 1 is negated and the external phase `φ` is applied to column 0 — those two steps are a convention fix-up so the error model reduces exactly to the original ideal MZI at `ε=0`. See `reports/neurophox_integration_report.md` and `reports/beamsplitter_math_derivation.md`.
- `ryser_permanent` is a fully vectorized 2^n-bitmask Ryser formula, so it is differentiable — this is what lets `routing.py` backprop through *quantum* probabilities.
- `propagate_fock` enumerates the full output Fock basis in Python, then uses `lax.map` (not `vmap`) over permanents to bound memory.
- `x64 is enabled globally` at import (`jax.config.update("jax_enable_x64", True)`); everything is complex128/float64.

**`decompose/pnn.py` — NumPy/SymPy Clements decomposition.** Self-contained, no JAX. `decompose_clements(U, block='mzi')` returns `(phis, thetas, alphas)` indexed `[mode_row, layer_pair]`. Critical convention mismatch when mapping into the engine (see `GUI._apply_unitary_decomposition`): **pnn's `theta` is half the engine's `theta`** (`engine_theta = 2 * pnn_theta`), and pnn's column index is `engine_col_idx // 2` with row index `mzi['mode_top']`. Getting this wrong silently produces a wrong unitary.

**`src/routing.py` — phase retrieval / inverse design.** Two independent strategies:
1. *Analytical* (`generate_routing_unitaries`): builds the exact solution family `U_k = V_out · diag(1, U_sub_k) · V_in†` for `U|ψ_in⟩ = |ψ_out⟩`, then hands each `U_k` to `decompose_clements`. Exact, no optimization. Rationale in `reports/phase_retrieval_plan.md`.
2. *Gradient-based* (`optimize_*_vmap`): `_run_vmap_optimization` runs N Adam restarts in parallel via `jit(vmap(...))` over a `lax.scan` inner loop, then dedupes solutions modulo 2π and sorts by loss. Restart/iteration counts are module-level constants chosen at import time from `jax.default_backend()` (GPU gets ~10x the budget). Coherent, incoherent, and quantum variants differ only in `loss_fn`.

**`src/gui.py` — Tkinter view + all mutable state.** Owns `self.phases[mzi_id] = {'theta', 'phi'}` and converts to engine arrays via `_get_phase_arrays()`. Changing `n_modes` destroys and rebuilds the whole widget tree plus a fresh `Engine`. Keyboard shortcuts on the selected MZI: `a` = bar, `s` = 50:50, `d` = cross.

**Hardware calibration path.** `node-isolation/8-mode-autocal-20260209.json` is a digital twin of a real 8-mode chip, keyed by heater id (`G4_theta`). Two uses:
- `Engine.load_calibration_errors()` derives per-MZI `ε` from fringe visibility: `ε² = (C - A) / (C + A)` where `A`/`C` are the fitted phase-response amplitude and offset.
- `tests/test_calibration_mapping.py` shows the full forward path: `parallel_nullification` → `apply_phase_constraints` (fold θ into [0, π) by flipping φ, minimizing heater power) → `phase = ω·P` → invert the thermo-optic resistance polynomial `P = c·I² + a·I⁴ + d·I⁶` for the required DAC current.

## Conventions

- Anything called inside a loss function or `vmap` must stay pure JAX; NumPy/SymPy live only in `decompose/pnn.py` and host-side setup.
- `reports/*.md` are the design record for the physics and algorithms — consult them before changing the MZI transfer matrix, the ε-from-visibility formula, or the phase retrieval approach.
- `unitary/*.npy` are generated artifacts, loadable in the GUI individually or as a folder to cycle through.
