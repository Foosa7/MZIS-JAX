# Estimating Beamsplitter Error from Node Isolation Calibration

This report details the mathematical foundation used to automatically extract individual Mach-Zehnder Interferometer (MZI) beamsplitter errors ($\epsilon$) from the node isolation dataset collected via the Alexiev et al. (2021) algorithm.

## 1. Mathematical Model of an Imperfect MZI

An ideal directional coupler (beamsplitter) splits light perfectly 50:50. In our model, we define a fabrication defect $\epsilon$ that breaks this symmetry. Following the physics model:
*   **Reflectivity (Bar state):** $R = \frac{\sqrt{1+\epsilon}}{\sqrt{2}}$
*   **Transmissivity (Cross state):** $T = \frac{i\sqrt{1-\epsilon}}{\sqrt{2}}$

The transfer matrix for a full MZI is constructed by multiplying the matrices of the two beamsplitters with an internal phase shift $\theta$ between them:
$$ U_{MZI} = BS(\epsilon) \cdot \begin{pmatrix} e^{i\theta} & 0 \\ 0 & 1 \end{pmatrix} \cdot BS(\epsilon) $$

If we solve for the transmission intensity of the Bar port ($T_{bar} = |U_{00}|^2$), the algebra simplifies remarkably:
$$ T_{bar} = \frac{1}{2} \left[ 1 + \epsilon^2 - (1-\epsilon^2)\cos\theta \right] $$

## 2. Theoretical Extinction & Visibility

By analyzing the transmission equation, we can find the maximum and minimum light that the MZI can possibly output during a full phase sweep:
*   **Max Transmission (when $\cos\theta = -1$):** 
    $$ T_{max} = \frac{1}{2} [1 + \epsilon^2 - (1-\epsilon^2)(-1)] = \mathbf{1.0} $$
*   **Min Transmission (when $\cos\theta = 1$):** 
    $$ T_{min} = \frac{1}{2} [1 + \epsilon^2 - (1-\epsilon^2)(1)] = \mathbf{\epsilon^2} $$

An ideal MZI ($\epsilon=0$) drops perfectly to $0.0$ power, giving an infinite extinction ratio. A defective MZI ($\epsilon > 0$) "bottoms out" at $\epsilon^2$ because the unequal power split prevents perfect destructive interference.

## 3. Mapping to Node Isolation Data

The node isolation algorithm measures output optical power as a function of electrical heating and fits it to a cosine curve:
$$ P_{out} = C - A \cos(\theta) $$
Where **$C$** is the Offset and **$A$** is the Amplitude.

Relating this to our transmission equation (and allowing for a global scaling factor $K$ representing laser input power and chip insertion loss):
*   $C = K \cdot \frac{T_{max} + T_{min}}{2} = K \cdot \frac{1 + \epsilon^2}{2}$
*   $A = K \cdot \frac{T_{max} - T_{min}}{2} = K \cdot \frac{1 - \epsilon^2}{2}$

## 4. Extracting $\epsilon$ (The brilliant part)

Because both the Amplitude ($A$) and Offset ($C$) suffer equally from the unknown global insertion loss ($K$), taking their ratio perfectly cancels $K$ out. This ratio is known as the **Fringe Visibility ($V$)**:

$$ V = \frac{A}{C} = \frac{1 - \epsilon^2}{1 + \epsilon^2} $$

Using basic algebra to isolate $\epsilon^2$:
$$ V(1 + \epsilon^2) = 1 - \epsilon^2 $$
$$ V + V\epsilon^2 = 1 - \epsilon^2 $$
$$ \epsilon^2(1 + V) = 1 - V $$
$$ \epsilon^2 = \frac{1 - V}{1 + V} $$

Substituting the Visibility back into the formula yields the exact equation implemented in our JAX Engine:
$$ \epsilon = \sqrt{\frac{C - A}{C + A}} $$

## Conclusion
By simply reading the `amplitude` ($A$) and `offset` ($C$) parameters for each node from your `8-mode-autocal-20260209.json`, the digital twin calculates this exact formula. It perfectly characterizes the structural beamsplitter defect of every single MZI on your chip without needing to know anything about the chip's overarching insertion losses!
