# Neurophox Integration Report: Phase Flow & Calibration

Following up on the beamsplitter error modeling integration, we have successfully ported the remaining high-value hardware calibration components from the `neurophox` repository directly into the `Engine` class in `src/engine.py`. These additions maintain the pure JAX standard for extreme performance while enhancing the realism of the digital twin.

## 1. JAX-Native Phase Flow & Constraints (`apply_phase_constraints`)

A core challenge in experimental photonics is dealing with unconstrained matrix decompositions. The raw math often requests heater phase shifts that are negative ($\phi < 0$) or extremely large ($\theta > 2\pi$), which thermo-optic elements physically cannot produce, or doing so would risk hardware degradation (micro-cracks in heaters).

We implemented the JAX-native `apply_phase_constraints` method which acts similarly to neurophox's `grid_common_mode_flow` to strictly enforce DAC heater boundaries.
*   **Modulo Normalization**: It mathematically guarantees all phases $\theta$ and $\phi$ strictly map within $[0, 2\pi)$.
*   **Power-Saving Phase Flipping**: If $\theta$ demands a high voltage placing it in $[\pi, 2\pi)$, the algorithm automatically applies a $\pi$ phase shift to $\phi$ and reflects $\theta$ to $[0, \pi)$. Due to MZI symmetries, this outputs the exact same unitary transformation but drastically reduces the required maximum thermal heater power across the chip, enhancing heater lifetime and reducing crosstalk.
*   **Vectorized**: Using `jnp.where` and `jnp.mod`, this calibration step is executed natively on the GPU/TPU instantly for the entire mesh in parallel.

## 2. Parallel Nullification Auto-Calibration (`parallel_nullification`)

To truly use the simulation as a digital twin of your physical optical chips, the engine must simulate how the hardware is calibrated in the lab. `neurophox` implements a `parallel_nullification` routine that replicates the procedure of minimizing cross-port optical power to find the "bar" state.

We ported this into the `Engine.parallel_nullification` static method:
*   **MZI Self-Calibration**: Given a set of complex optical input amplitudes (`amps_in`), the algorithm mathematically deduces the exact $\theta$ and $\phi$ required to "nullify" the cross port and route all power perfectly to the bar port.
*   **Error-Aware**: This new nullification algorithm is deeply integrated with the beamsplitter error matrices added in the previous update. When calculating the necessary $\theta$ required to balance out unequal input powers, it factors in the left beamsplitter's explicit error bias ($\epsilon_l$) via perturbed transmissivity $\sqrt{1 \pm \epsilon}$, rather than assuming a perfect 50:50 splitting.
*   **JIT Compiled**: Being decorated with `@jit`, this function can be used dynamically within larger `lax.scan` loops to simulate an automatic layer-by-layer chip calibration sweep in a fraction of a second.

---
**Summary of Engine Additions (`src/engine.py`)**
*   `Engine.apply_phase_constraints(thetas, phis)`
*   `Engine.parallel_nullification(amps_in, e_l, e_r)`
