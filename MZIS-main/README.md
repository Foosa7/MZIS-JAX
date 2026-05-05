# MZI Simulator (MZIS)

An interactive simulation tool for programmable photonic integrated circuits. This application models an interferometer mesh of Mach-Zehnder Interferometers (MZIs) to simulate both classical light propagation and quantum multi-photon interference.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KTH-QUIP/MZIS.git
    cd MZIS
    ```

2.  **Install dependencies:**
    This project requires Python 3.10+ and the following packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Dependencies: `numpy`, `matplotlib`)*
    
    > **Note:** The interface uses `tkinter`, which is included with most standard Python installations. If you are on Linux and encounter an error, you may need to install it separately (e.g., `sudo apt-get install python3-tk`).

## Usage

To start the simulator, simply run the main script from the root directory:

```bash
python main.py
```

### How to Use

1.  **Input Configuration:** Use the panel on the left to inject photons into specific input ports.
2.  **Tune the Mesh:** 
    *   **Select:** Click on any MZI node (the crossing points) in the central mesh view.
    *   **Adjust:** Use the sliders in the left panel to change the internal phase ($\theta$) or external phase ($\phi$).
    *   **Presets:** Use buttons like "Bar", "50:50", or "Cross" for quick configuration.
3.  **Analyze:** 
    *   Observe the brightness of the paths to see classical power distribution.
    *   Watch the bar chart on the right for the probability distribution of quantum output states.

## How it Works

The core engine uses a unitary matrix representation of the linear optical network.
*   **Classical:** Propagates the input vector through the matrix layers to compute intensities.
*   **Quantum:** Computes the permanent of the scattering submatrices to determine the transition probabilities for specific Fock states, capturing quantum interference effects like the Hong-Ou-Mandel effect.
