# Neurophox Integration Report

This report summarizes the codebase improvements and features borrowed from the `neurophox` repository into our core JAX photonic simulation engine. 

## 1. JAX-Native Beamsplitter Error Modeling (Integrated)

Real-world photonic meshes suffer from fabrication variances where directional couplers (beamsplitters) do not perfectly split light 50:50. `neurophox` accounts for this by defining a beamsplitter error $\epsilon$ and scaling the transfer matrices.

We have successfully integrated this behavior directly into `src/engine.py` while keeping the entire simulation pipeline in **pure JAX** (avoiding any NumPy overhead).

**Key Changes Made:**
*   Modified the `Engine` class constructor to accept a `bs_error` parameter, allowing either a scalar global error or a tuple of `(e_l, e_r)` arrays representing individual left/right errors for every single MZI in the mesh.
*   Rewrote the inner loop of `_build_layer_matrix` and `_compute_full_unitary` to construct the unitary matrix using the explicit physical model: $U_{MZI} = BS(\epsilon_r) \cdot Phase \cdot BS(\epsilon_l)$.
*   Used JAX primitives (`jnp.sqrt`, `jnp.exp`) to dynamically compute the perturbed transmissivity and reflectivity terms:
    *   Reflectivity (Bar state): $\sqrt{1+\epsilon} / \sqrt{2}$
    *   Transmissivity (Cross state): $i\sqrt{1-\epsilon} / \sqrt{2}$
*   **Backwards Compatibility:** We carefully mapped the neurophox theoretical formulation back to our existing engine's coordinate convention. By strategically negating the second column of the transfer matrix and mapping the external phase $\phi$ to the top input port, our non-ideal JAX MZI perfectly reduces to your original ideal MZI when $\epsilon=0$. This ensures your HOM Dip and Random Boson Sampling GUI demos continue to work seamlessly!

## 2. Phase Flow & Normalization (Planned/Available for Borrowing)

The `neurophox` decomposition module (`decompositions.py`) features a highly robust algorithm called `grid_common_mode_flow`. 

**Why it's useful:**
When performing Clements decompositions, the raw mathematical algorithm often produces phase shifts ( $\phi$ and $\theta$ ) that span across vast ranges (e.g., negative phases, or values far above $2\pi$). Physical thermo-optic heaters can only apply *positive* phase shifts, and pushing them too hard physically degrades the chip.
The `grid_common_mode_flow` algorithm pushes these "common mode" phases forward through the mesh layers sequentially until they accumulate at the very end of the chip as a single vector of output phase shifts (referred to as $\gamma$).

**How we will integrate it in JAX:**
Currently, `decompose/pnn.py` computes the raw Clements parameters in NumPy. We can port the `grid_common_mode_flow` logic to execute in JAX as an optimization pass applied right before hardware mapping, strictly constraining all physical MZIs to $0 \le \phi < 2\pi$.

## 3. Parallel Nullification (Auto-Calibration)

Neurophox contains a `parallel_nullification` algorithm mimicking the experimental procedure for calibrating a mesh layer-by-layer by minimizing power on alternating ports. We can translate this routine into a JAX differentiable function in the future, allowing us to simulate realistic calibration routines on top of the digital twin we are building.

---
**Summary of Engine Changes (`src/engine.py`)**
*   `__init__` now accepts `bs_error`
*   `_build_layer_matrix` refactored to multiply three sub-matrices representing the physical MZI components.
*   `_compute_full_unitary` updated to slice and pass dynamic `e_l` and `e_r` arrays into the JAX `lax.scan` bodies.
