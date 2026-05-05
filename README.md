# JAX Photonic Engine

A high-performance, JAX-based photonic simulation engine and interactive visualizer for Mach-Zehnder Interferometer (MZI) meshes.

## Features
* **Pure Functional Core:** The simulation engine is written entirely in JAX, allowing for JIT compilation, automatic differentiation, and GPU acceleration.
* **Quantum & Classical Modes:** Simulates coherent classical power flow and quantum state propagation (including Hong-Ou-Mandel interference via Ryser's permanent algorithm).
* **Interactive GUI:** Real-time visualization of the MZI mesh with interactive phase controls.
* **Unitary & Switching Decomposition:** Instantly generate or import arbitrary target unitaries or permutation (switching) matrices, automatically decomposing them into Clement grid phase settings using `pnn.py`.

## Repository Structure
* `src/`: Core simulation logic (`engine.py`) and UI (`gui.py`).
* `decompose/`: Algorithms for Clements matrix decomposition.
* `scripts/`: Tools to generate random unitary and switching arrays.
* `unitary/`: Directory containing generated `.npy` matrix files.
* `tests/`: Scripts for testing interference (e.g., HOM dip).

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the interactive visualization app:
```bash
python main.py
```
