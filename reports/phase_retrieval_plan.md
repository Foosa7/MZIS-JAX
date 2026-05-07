# Implementation Report & Plan: Photonic Phase Retrieval and Routing

## 1. Executive Summary
The objective is to compute a set of $n$ unitary matrices (and subsequently their hardware phase configurations) that satisfy a specific routing constraint: mapping a known input optical state vector $|\psi_{in}\rangle$ to a known output state vector $|\psi_{out}\rangle$. 

Because a single input-output constraint $U |\psi_{in}\rangle = |\psi_{out}\rangle$ only removes $2N-1$ degrees of freedom from an $N \times N$ unitary matrix (which has $N^2$ degrees of freedom), there is an infinite, continuous manifold of valid unitaries. The goal is to compute the first $n$ valid configurations to present to the user or experimentalist.

## 2. Mathematical Formulation
Given normalized complex state vectors $|\psi_{in}\rangle$ and $|\psi_{out}\rangle$, we seek $U \in U(N)$ such that:
$$ U |\psi_{in}\rangle = |\psi_{out}\rangle $$

The set of all solutions forms a coset of the $U(N-1)$ subgroup. We can systematically sample this space analytically without relying on iterative optimization, ensuring we find mathematically exact, distinct configurations instantly.

## 3. Recommended Approach: Analytical Subspace Sampling
This method uses basis transformations to "hide" the unconstrained dimensions, allowing us to rapidly generate an arbitrary number of valid unitaries.

### Implementation Steps:
**Step 1: Basis Generation**
Find a unitary transformation $V_{in}$ that rotates the standard computational basis state $|0\rangle$ (e.g., light entirely in port 0) into $|\psi_{in}\rangle$. Similarly, find $V_{out}$ mapping $|0\rangle \to |\psi_{out}\rangle$.
*(This can be computed using QR decomposition by placing the state vector in the first column of a random matrix).*

**Step 2: Subspace Injection**
Any unitary matrix that routes $|0\rangle \to |0\rangle$ must have the block diagonal form:
$$ W = \begin{pmatrix} 1 & 0 \\ 0 & U_{sub} \end{pmatrix} $$
where $U_{sub}$ is any $(N-1) \times (N-1)$ unitary. By sampling random $U_{sub}$ matrices (e.g., from the Haar measure), we generate diverse valid $W$ matrices.

**Step 3: Construct Routing Unitaries**
The complete family of solutions mapping $|\psi_{in}\rangle \to |\psi_{out}\rangle$ is precisely parameterized by:
$$ U_k = V_{out} \begin{pmatrix} 1 & 0 \\ 0 & U_{sub, k} \end{pmatrix} V_{in}^\dagger $$
We loop this $n$ times to generate $\{U_1, U_2, \dots, U_n\}$.

**Step 4: Hardware Decomposition**
We pass each generated $U_k$ into your existing `decompose_clements(U_k, block='mzi')` from `decompose/pnn.py` to extract the corresponding hardware MZI phases ($\theta$, $\phi$).

> [!TIP]
> **Why this is optimal:** It's $O(N^3)$ and perfectly exact. There is no risk of gradient descent getting stuck in local minima, and it guarantees 100% fidelity mathematically before even applying it to the simulated mesh. A working prototype script for this math was just successfully verified in `scripts/phase_retrieval_demo.py`.

## 4. Alternative Approach: Gradient-Based Optimization in JAX
If you need to route light under physical hardware constraints (e.g., bounding the total heater power, avoiding certain broken MZIs, or adding dispersion/loss penalties), we can utilize your existing JAX `Engine`.

### Implementation Steps:
1. **Define Loss Function:** Compute the fidelity of routing using the `Engine.compute_full_unitary(thetas, phis)`:
   $$ \mathcal{L}(\theta, \phi) = 1 - \left| \langle \psi_{out} | U(\theta, \phi) | \psi_{in} \rangle \right|^2 $$
2. **Add Penalties:** Optionally add penalties for $\theta, \phi$ values that exceed $2\pi$ or require excessive current.
3. **Optimization Loop:** Use `jax.grad` and an optimizer like `optax.adam`.
4. **Batch Execution:** Run the optimizer with $n$ different random seeds (random initial $\theta, \phi$ arrays). JAX's `vmap` allows us to parallelize this and solve all $n$ configurations simultaneously on the GPU/CPU.

## 5. Execution Plan
To implement this in the codebase:
1. **Create `src/routing.py`:** Add a `StateRouter` class containing both the `generate_analytical_unitaries` method and the `optimize_constrained_routing` method.
2. **Integrate with Digital Twin:** Add a pipeline that connects the generated unitaries to the `PhotonicTwin` or directly visualizes them on the GUI via `window1.py`.
3. **Batch Export:** Enable the user to save the $n$ distinct generated unitaries into a folder of JSON files, directly compatible with your new "Unitary Folder Import" tool.
