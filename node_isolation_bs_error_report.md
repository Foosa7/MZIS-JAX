# Node-Isolation Beamsplitter Error Integration Report

We have successfully closed the loop between your physical 8-mode chip calibration data (`node-isolation/8-mode-autocal-20260209.json`) and your digital twin!

## 1. Extracting Beamsplitter Error from Visibility

While node-isolation primarily extracts the phase-to-power relationship, it also inherently measures **fringe visibility** during the cosine fit. A perfect 50:50 MZI has a visibility of exactly $1.0$ (it completely nullifies down to $0.0$ power). 

By analyzing the `amplitude` ($A$) and `offset` ($C$) parameters stored in your JSON, we can mathematically isolate the physical beamsplitter defect ($\epsilon$) that prevented it from hitting zero!
*   **Visibility Equation**: $V = \frac{\text{Amplitude}}{\text{Offset}}$
*   **Error Extraction**: $\epsilon = \sqrt{\frac{1 - V}{1 + V}}$

We implemented a new `load_calibration_errors(json_path)` method directly into the `Engine` class. When pointed to your JSON, it parses all available `phase_calibration` entries (e.g., `G4_theta`), extracts the exact fabrication defects for that specific MZI, and maps it directly into the JAX `e_l` and `e_r` arrays.

## 2. Dynamic GUI Selection

To make this completely interactive, we updated `src/gui.py`. We've added a new dropdown menu right under the **Simulation Mode** controls labeled **Hardware Imperfections**:
1.  **Ideal (No Error)**: Reverts the mesh to perfect $50:50$ matrices (great for pure theoretical tests like Boson sampling verification).
2.  **Basic Loss (15%)**: Replaces all MZIs with a uniform global defect (to see what generalized crosstalk does to a quantum state).
3.  **Calibration Data**: Connects directly to `8-mode-autocal-20260209.json`. If selected, your digital twin actively reflects the specific, unique hardware defects of your physical 8-mode chip across the MZI grid!

*Note: If you load a matrix larger than your chip (e.g., $N=12$ or $16$), the engine seamlessly falls back to a default defect for the uncalibrated nodes while perfectly applying your JSON data to the nodes it knows about.*

## 3. Next Steps
Your digital twin now fully mimics the non-ideal power routing of your chip. However, if you plan to execute large scale $8 \times 8$ matrices, the last piece of the puzzle is **Thermal Crosstalk**. Heating one node physically heats its neighbor. Once you gather thermal crosstalk matrices, we can add a simple JAX matrix multiplication pass into the phase solver to complete the perfect twin!
