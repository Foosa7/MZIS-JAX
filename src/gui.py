import tkinter as tk
from tkinter import ttk
import numpy as np
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .engine import Engine
from .routing import StateRouter

class GUI:
    def __init__(self, root, n_modes=8):
        """Initializes the GUI and simulation engine."""
        self.root = root
        self.n_modes = n_modes
        self.root.title(f"Interferometer Simulator ({n_modes} Modes)")
        self.root.geometry("1400x900")
        
        # color palette
        self.colors = {
            'bg': '#1e1e1e',
            'panel': '#252526',
            'fg': '#ffffff',
            'accent': '#00ffcc', 
            'grid_lines': '#3e3e42',
            'waveguide_off': '#404040',
            'highlight': '#007acc'
        }
        
        self.root.configure(bg=self.colors['bg'])

        self.engine = Engine(n_modes=self.n_modes, bs_error=0.0)
        self.selected_mzi = None
        self.input_vars = [0] * n_modes
        self.sim_mode = tk.StringVar(value="quantum")
        self.pr_active = False
        self.target_vars = [tk.DoubleVar(value=0.0) for _ in range(n_modes)]
        self.loaded_unitaries = []
        self.unitary_files = []
        self.current_unitary_idx = -1
        
        self.phases = {}
        for mid in self.engine.mzi_ids:
            self.phases[mid] = {'theta': float(jnp.pi), 'phi': 0.0}
        
        self._setup_styles()
        self._create_layout()
        
        self.root.update() 
        self._draw_mesh()
        self._update_simulation()

    def _setup_styles(self):
        """Configures the visual styles for Tkinter widgets."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # widget styling
        style.configure("TFrame", background=self.colors['bg'])
        style.configure("Panel.TFrame", background=self.colors['panel'])
        style.configure("TLabel", background=self.colors['bg'], foreground=self.colors['fg'], font=("Arial", 10))
        style.configure("Panel.TLabel", background=self.colors['panel'], foreground=self.colors['fg'])
        style.configure("Header.TLabel", font=("Arial", 12, "bold"), foreground=self.colors['accent'], background=self.colors['panel'])
        
        # button styling
        style.configure("TButton", 
                        background="#333333", 
                        foreground="white", 
                        borderwidth=0, 
                        padding=5)
        style.map("TButton", 
                  background=[('active', '#4d4d4d'), ('pressed', '#2d2d2d')])
        
        # scrollbar
        style.configure("Vertical.TScrollbar",
                        background="#333333",
                        troughcolor=self.colors['panel'],
                        borderwidth=0,
                        width=10,
                        arrowsize=0,
                        arrowcolor=self.colors['panel'])
        style.map("Vertical.TScrollbar",
                  background=[('active', '#555555'), ('pressed', '#444444')])
        
        # separator
        style.configure("TSeparator", background=self.colors['grid_lines'])

    def _create_layout(self):
        """Constructs the main interface layout including control panels and canvas."""
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        control_panel = ttk.Frame(self.main_container, style="Panel.TFrame", width=340)
        control_panel.pack(side=tk.LEFT, fill=tk.Y)
        control_panel.pack_propagate(False) # Force width
        
        # Make the control panel scrollable
        self.panel_canvas = tk.Canvas(control_panel, bg=self.colors['panel'], highlightthickness=0)
        panel_scrollbar = ttk.Scrollbar(control_panel, orient="vertical", command=self.panel_canvas.yview)
        
        pad_frame = ttk.Frame(self.panel_canvas, style="Panel.TFrame")
        
        # Configure scroll region when inner frame changes size
        pad_frame.bind(
            "<Configure>",
            lambda e: self.panel_canvas.configure(scrollregion=self.panel_canvas.bbox("all"))
        )
        
        # Window inside canvas (anchor nw, width set to fill space minus scrollbar)
        self.panel_canvas.create_window((0, 0), window=pad_frame, anchor="nw", width=310)
        self.panel_canvas.configure(yscrollcommand=panel_scrollbar.set)
        
        self.panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 0), pady=15)
        panel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=15)

        # Mouse wheel / trackpad scrolling bindings
        def _on_mousewheel(event):
            # Linux X11 discrete scroll buttons
            if event.num == 4:
                self.panel_canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self.panel_canvas.yview_scroll(3, "units")
            else:
                # Windows / macOS / trackpad smooth scroll
                delta = getattr(event, 'delta', 0)
                if delta:
                    self.panel_canvas.yview_scroll(int(-delta / 60), "units")

        def _bind_scroll(event):
            self.panel_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.panel_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.panel_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_scroll(event):
            self.panel_canvas.unbind_all("<MouseWheel>")
            self.panel_canvas.unbind_all("<Button-4>")
            self.panel_canvas.unbind_all("<Button-5>")

        # Bind to the entire control panel so scrolling works over any child widget
        control_panel.bind('<Enter>', _bind_scroll)
        control_panel.bind('<Leave>', _unbind_scroll)

        # Chip size buttons
        ttk.Label(pad_frame, text="Chip Architecture", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        size_frame = ttk.Frame(pad_frame, style="Panel.TFrame")
        size_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.size_buttons = {}
        for n in [4, 8, 12, 16]:
            bg = self.colors['accent'] if n == self.n_modes else "#333"
            fg = "black" if n == self.n_modes else "white"
            btn = tk.Button(size_frame, text=f"N={n}", bg=bg, fg=fg, bd=0,
                            font=("Arial", 10, "bold"), command=lambda n=n: self._set_modes(n))
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            self.size_buttons[n] = btn

        ttk.Label(pad_frame, text="Simulation Mode", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        mode_frame = ttk.Frame(pad_frame, style="Panel.TFrame")
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.btn_q = tk.Button(mode_frame, text="Quantum", bg=self.colors['accent'], fg="black", bd=0, 
                               font=("Arial", 10, "bold"), command=lambda: self._set_sim_mode("quantum"))
        self.btn_q.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        self.btn_c = tk.Button(mode_frame, text="Classical", bg="#333", fg="white", bd=0, 
                               font=("Arial", 10, "bold"), command=lambda: self._set_sim_mode("classical"))
        self.btn_c.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 2))
        
        self.btn_pr = tk.Button(mode_frame, text="Phase Retrieval", bg="#333", fg="white", bd=0, 
                               font=("Arial", 10, "bold"), command=self._toggle_pr_mode)
        self.btn_pr.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 0))
        
        # Initialize button colors
        if self.sim_mode.get() == "classical":
            self.btn_q.config(bg="#333", fg="white")
            self.btn_c.config(bg=self.colors['accent'], fg="black")

        # Hardware Errors
        ttk.Label(pad_frame, text="Hardware Imperfections", style="Header.TLabel").pack(anchor="w", pady=(15, 5))
        hw_frame = ttk.Frame(pad_frame, style="Panel.TFrame")
        hw_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.error_mode = tk.StringVar(value="Ideal (No Losses)")
        self.error_dropdown = tk.OptionMenu(hw_frame, self.error_mode, 
                                            "Ideal (No Losses)", "Basic Losses (15%)", "Calibration Data",
                                            command=self._on_error_change)
        self.error_dropdown.config(bg="#333", fg="white", activebackground="#444", activeforeground="white",
                                   bd=0, highlightthickness=0, font=("Arial", 10, "bold"))
        self.error_dropdown["menu"].config(bg="#333", fg="white", activebackground=self.colors['accent'], activeforeground="black", font=("Arial", 10))
        self.error_dropdown.pack(fill=tk.X, pady=5)

        ttk.Label(pad_frame, text="Light Inputs", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
        
        input_grid = ttk.Frame(pad_frame, style="Panel.TFrame")
        input_grid.pack(fill=tk.X)
        
        self.input_labels = []
        for i in range(self.n_modes):
            row = ttk.Frame(input_grid, style="Panel.TFrame")
            row.pack(fill=tk.X, pady=2)
            
            ttk.Label(row, text=f"Port {i+1}", style="Panel.TLabel", width=8).pack(side=tk.LEFT)
            
            # minus button
            btn_minus = tk.Button(row, text="-", width=3, bg="#333", fg="white", bd=0,
                                  activebackground="#444", activeforeground="white",
                                  command=lambda idx=i: self._change_input(idx, -1))
            btn_minus.pack(side=tk.LEFT, padx=2)
            
            # value label
            lbl = tk.Label(row, text="0", width=4, bg="#1e1e1e", fg=self.colors['accent'], font=("Arial", 11, "bold"))
            lbl.pack(side=tk.LEFT, padx=2)
            self.input_labels.append(lbl)
            
            # plus button
            btn_plus = tk.Button(row, text="+", width=3, bg="#333", fg="white", bd=0,
                                 activebackground="#444", activeforeground="white",
                                 command=lambda idx=i: self._change_input(idx, 1))
            btn_plus.pack(side=tk.LEFT, padx=2)

        ttk.Separator(pad_frame, orient='horizontal').pack(fill='x', pady=20)

        self.dynamic_frame = ttk.Frame(pad_frame, style="Panel.TFrame")
        self.dynamic_frame.pack(fill=tk.X)

        self.target_frame = ttk.Frame(self.dynamic_frame, style="Panel.TFrame")
        
        ttk.Label(self.target_frame, text="Target Outputs (Phase Retrieval)", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
        for i in range(self.n_modes):
            f = ttk.Frame(self.target_frame, style="Panel.TFrame")
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=f"Port {i+1}", style="Panel.TLabel", width=8).pack(side=tk.LEFT)
            scale = tk.Scale(f, from_=0, to=1.0, variable=self.target_vars[i], orient=tk.HORIZONTAL, 
                             bg=self.colors['panel'], fg="white", troughcolor="#111",
                             activebackground=self.colors['accent'], highlightthickness=0, bd=0,
                             resolution=0.01)
            scale.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        btn_compute = tk.Button(self.target_frame, text="Compute Routing", bg=self.colors['accent'], fg="black",
                                font=("Arial", 11, "bold"), bd=0, activebackground="#00cc99",
                                command=self._on_target_change)
        btn_compute.pack(fill=tk.X, pady=(10, 5), ipady=6)
            
        export_frame = ttk.Frame(self.target_frame, style="Panel.TFrame")
        export_frame.pack(fill=tk.X, pady=(0, 5))
        btn_export = ttk.Button(export_frame, text="Export Current Unitary", command=self._export_current_unitary)
        btn_export.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        btn_export_top3 = ttk.Button(export_frame, text="Export Top 3", command=self._export_top3_unitaries)
        btn_export_top3.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        
        btn_reset_pr = ttk.Button(self.target_frame, text="Reset to Identity", command=self._demo_clear)
        btn_reset_pr.pack(fill=tk.X, pady=(0, 5))

        self.mzi_frame = ttk.Frame(self.dynamic_frame, style="Panel.TFrame")
        self.mzi_frame.pack(fill=tk.X)
        self.target_frame.pack_forget()
        
        ttk.Label(self.mzi_frame, text="MZI Config", style="Header.TLabel").pack(anchor="w")
        self.lbl_selected = ttk.Label(self.mzi_frame, text="Select an MZI on the mesh", 
                                      foreground="#888", style="Panel.TLabel")
        self.lbl_selected.pack(pady=(5, 15), anchor="w")

        self.theta_var = tk.DoubleVar()
        self.phi_var = tk.DoubleVar()
        
        def create_slider(parent, label, var, r_max):
            """Creates a labeled slider widget for phase adjustment."""
            f = ttk.Frame(parent, style="Panel.TFrame")
            f.pack(fill=tk.X, pady=5)
            ttk.Label(f, text=label, style="Panel.TLabel").pack(anchor="w")
            scale = tk.Scale(f, from_=0, to=r_max, variable=var, orient=tk.HORIZONTAL, 
                             bg=self.colors['panel'], fg="white", troughcolor="#111",
                             activebackground=self.colors['accent'], highlightthickness=0, bd=0,
                             resolution=0.01, command=lambda x: self._on_param_change())
            scale.pack(fill=tk.X)
            
        create_slider(self.mzi_frame, "Internal phase (θ) - Splitting (xπ)", self.theta_var, 2.0)
        create_slider(self.mzi_frame, "External phase (φ) - Phase Shift (xπ)", self.phi_var, 2.0)

        # preset buttons
        ttk.Label(self.mzi_frame, text="Quick Presets", style="Panel.TLabel").pack(anchor="w", pady=(15, 5))
        btn_frame = ttk.Frame(self.mzi_frame, style="Panel.TFrame")
        btn_frame.pack(fill=tk.X)
        
        # helper to style buttons
        def mk_btn(txt, t, p):
            """Creates a preset button with specific theta and phi values."""
            b = ttk.Button(btn_frame, text=txt, command=lambda: self._set_preset(t, p))
            b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            
        mk_btn("Bar", 1.0, 0.0)
        mk_btn("50:50", 0.5, 0.0)
        mk_btn("Cross", 0.0, 0.0)

        # Demos
        ttk.Separator(pad_frame, orient='horizontal').pack(fill='x', pady=20)
        
        self.demo_frame = ttk.Frame(pad_frame, style="Panel.TFrame")
        self.demo_frame.pack(fill=tk.X)
        ttk.Label(self.demo_frame, text="Demos", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        
        btn_hom = ttk.Button(self.demo_frame, text="HOM Dip (2 Photons)", command=self._demo_hom)
        btn_hom.pack(fill=tk.X, pady=2)
        
        btn_rand = ttk.Button(self.demo_frame, text="Random Boson Sampling", command=self._demo_random)
        btn_rand.pack(fill=tk.X, pady=2)
        
        decomp_frame = ttk.Frame(self.demo_frame, style="Panel.TFrame")
        decomp_frame.pack(fill=tk.X, pady=2)
        
        btn_haar = ttk.Button(decomp_frame, text="Haar Random", command=self._demo_unitary_decomposition)
        btn_haar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        btn_switch = ttk.Button(decomp_frame, text="Switching Matrix", command=self._demo_switching_decomposition)
        btn_switch.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        
        btn_import = ttk.Button(self.demo_frame, text="Import Unitary", command=self._import_unitary_decomposition)
        btn_import.pack(fill=tk.X, pady=2)
        
        btn_import_folder = ttk.Button(self.demo_frame, text="Import Unitary Folder", command=self._import_unitary_folder)
        btn_import_folder.pack(fill=tk.X, pady=2)
        
        btn_clear = ttk.Button(self.demo_frame, text="Reset to Identity", command=self._demo_clear)
        btn_clear.pack(fill=tk.X, pady=2)

        center_area = ttk.Frame(self.main_container, style="TFrame")
        center_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # mesh area container
        mesh_area = ttk.Frame(center_area, style="TFrame")
        mesh_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        # mesh canvas 
        self.canvas = tk.Canvas(mesh_area, bg=self.colors['bg'], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._draw_mesh)
        
        # Keyboard shortcuts for MZI presets
        self.root.bind("a", lambda e: self._set_preset(1.0, 0.0))   # Bar
        self.root.bind("s", lambda e: self._set_preset(0.5, 0.0))   # 50:50
        self.root.bind("d", lambda e: self._set_preset(0.0, 0.0))   # Cross
        
        # Unitary cycle controls below canvas
        self.cycle_frame = ttk.Frame(mesh_area, style="Panel.TFrame")
        self.cycle_frame.pack(fill=tk.X, pady=(10, 0))
        
        btn_prev = ttk.Button(self.cycle_frame, text="< Prev", command=self._prev_unitary)
        btn_prev.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.lbl_unitary = ttk.Label(self.cycle_frame, text="No unitaries loaded", style="Panel.TLabel", anchor="center")
        self.lbl_unitary.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        
        btn_next = ttk.Button(self.cycle_frame, text="Next >", command=self._next_unitary)
        btn_next.pack(side=tk.RIGHT, padx=5, pady=5)

        # graph 
        self.plot_frame = ttk.Frame(center_area, width=350, style="TFrame")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        self.plot_frame.pack_propagate(False) # Fix width
        
        self.fig = plt.Figure(figsize=(4, 8), facecolor=self.colors['bg'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['bg'])
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        plot_widget = self.canvas_plot.get_tk_widget()
        plot_widget.configure(bg=self.colors['bg'], highlightthickness=0)
        plot_widget.pack(fill=tk.BOTH, expand=True)

    def _set_sim_mode(self, mode):
        """Sets the simulation mode (quantum/classical) and updates UI."""
        self.sim_mode.set(mode)
        if mode == "quantum":
            self.btn_q.config(bg=self.colors['accent'], fg="black")
            self.btn_c.config(bg="#333", fg="white")
        else:
            self.btn_q.config(bg="#333", fg="white")
            self.btn_c.config(bg=self.colors['accent'], fg="black")
        self._update_simulation()
    
    def _toggle_pr_mode(self):
        """Toggles Phase Retrieval overlay on/off."""
        self.pr_active = not self.pr_active
        if self.pr_active:
            self.btn_pr.config(bg=self.colors['accent'], fg="black")
            self.mzi_frame.pack_forget()
            self.target_frame.pack(fill=tk.X)
            self.target_vars[0].set(1.0)
            for i in range(1, self.n_modes):
                self.target_vars[i].set(0.0)

        else:
            self.btn_pr.config(bg="#333", fg="white")
            self.target_frame.pack_forget()
            self.mzi_frame.pack(fill=tk.X)
            self._update_simulation()

    def _on_target_change(self):
        if not self.pr_active:
            return
            
        P_in = np.array(self.input_vars, dtype=np.float64)
        P_out = np.array([max(0, v.get()) for v in self.target_vars], dtype=np.float64)
        
        if np.sum(P_in) == 0:
            P_in[0] = 1.0
            
        n_photons = int(np.sum(P_in))
        active_inputs = np.sum(P_in > 0)
        is_quantum = self.sim_mode.get() == "quantum"
        
        if is_quantum and n_photons > 0:
            # Quantum-aware: optimize directly for Fock detection probabilities
            from .routing import _QUANTUM_RESTARTS, _QUANTUM_ITERS
            self.lbl_unitary.config(text=f"Quantum optimization ({n_photons} photon{'s' if n_photons > 1 else ''}, {_QUANTUM_RESTARTS}x{_QUANTUM_ITERS})...")
            self.root.update()
            
            input_occ = [int(x) for x in P_in]
            results = StateRouter.optimize_quantum_routing_vmap(
                self.engine, input_occ, P_out
            )
        elif active_inputs <= 1:
            # Classical single input: coherent field-level optimization
            from .routing import _CLASSICAL_RESTARTS, _CLASSICAL_ITERS
            self.lbl_unitary.config(text=f"Coherent optimization ({_CLASSICAL_RESTARTS}x{_CLASSICAL_ITERS})...")
            self.root.update()
            
            psi_in = np.sqrt(P_in).astype(np.complex128)
            norm = np.linalg.norm(psi_in)
            if norm > 0:
                psi_in /= norm
            else:
                psi_in[0] = 1.0
                
            psi_target = np.sqrt(P_out).astype(np.complex128)
            norm_t = np.linalg.norm(psi_target)
            if norm_t > 0:
                psi_target /= norm_t
            else:
                psi_target[0] = 1.0

            results = StateRouter.optimize_coherent_routing_vmap(self.engine, psi_in, psi_target)
        else:
            # Classical multi-input: incoherent power-level optimization
            from .routing import _CLASSICAL_RESTARTS, _CLASSICAL_ITERS
            self.lbl_unitary.config(text=f"Incoherent optimization ({_CLASSICAL_RESTARTS}x{_CLASSICAL_ITERS})...")
            self.root.update()
            results = StateRouter.optimize_incoherent_routing_vmap(self.engine, P_in, P_out)
        
        # Store all results as phase configurations (not unitaries to decompose)
        self._pr_results = results  # Keep raw phase results for export
        self.loaded_unitaries = []
        self.unitary_files = []
        
        for i, (thetas, phis, loss) in enumerate(results):
            U = self.engine.compute_full_unitary(thetas, phis)
            self.loaded_unitaries.append(np.asarray(U))
            self.unitary_files.append(f"Option {i+1} (Loss: {loss:.6f})")
            
        self.current_unitary_idx = 0
        self._apply_pr_result(0)
    
    def _apply_pr_result(self, idx):
        """Applies a phase retrieval result directly to the mesh (no Clements decomposition)."""
        if not hasattr(self, '_pr_results') or not self._pr_results:
            return
            
        thetas, phis, loss = self._pr_results[idx]
        
        # Apply phases directly to mesh
        for i, mid in enumerate(self.engine.mzi_ids):
            self.phases[mid]['theta'] = float(thetas[i])
            self.phases[mid]['phi'] = float(phis[i])
            if self.selected_mzi == mid:
                self.theta_var.set(float(thetas[i]) / float(np.pi))
                self.phi_var.set(float(phis[i]) / float(np.pi))
        
        # Compute actual output for feedback
        thetas_arr, phis_arr = self._get_phase_arrays()
        U = self.engine.compute_full_unitary(thetas_arr, phis_arr)
        
        P_in = np.array(self.input_vars, dtype=np.float64)
        if np.sum(P_in) == 0:
            P_in[0] = 1.0
            
        active_inputs = np.sum(P_in > 0)
        if active_inputs <= 1:
            psi_in = np.sqrt(P_in).astype(np.complex128)
            psi_in /= np.linalg.norm(psi_in)
            psi_out = np.asarray(U) @ psi_in
            P_actual = np.abs(psi_out)**2
        else:
            power_trans = np.abs(np.asarray(U))**2
            P_actual = power_trans @ P_in
        
        P_target = np.array([max(0, v.get()) for v in self.target_vars], dtype=np.float64)
        if np.sum(P_target) > 0:
            P_target = P_target * (np.sum(P_in) / np.sum(P_target))
        
        self.lbl_unitary.config(text=f"Option {idx+1}/{len(self._pr_results)} (Loss: {loss:.6f})")
        self._update_simulation()

    def _export_current_unitary(self):
        from tkinter import filedialog, messagebox
        import os
        
        thetas, phis = self._get_phase_arrays()
        U = self.engine.compute_full_unitary(thetas, phis)
        
        filepath = filedialog.asksaveasfilename(
            title="Save Unitary",
            defaultextension=".npz",
            filetypes=[("Numpy archive", "*.npz")],
            initialfile="retrieved_unitary.npz"
        )
        if filepath:
            try:
                np.savez(filepath, 
                         unitary=np.asarray(U),
                         thetas=np.asarray(thetas),
                         phis=np.asarray(phis))
                messagebox.showinfo("Export Successful", f"Saved to {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save: {e}")

    def _export_top3_unitaries(self):
        """Exports the top 3 best routing unitaries to .npy files."""
        from tkinter import filedialog, messagebox
        import os
        
        if not hasattr(self, '_pr_results') or not self._pr_results:
            messagebox.showwarning("No Results", "Run phase retrieval first.")
            return
            
        folder = filedialog.askdirectory(title="Select folder to save top 3 unitaries")
        if not folder:
            return
            
        top_n = min(3, len(self._pr_results))
        saved = []
        for i in range(top_n):
            thetas, phis, loss = self._pr_results[i]
            U = self.engine.compute_full_unitary(thetas, phis)
            fname = f"routing_rank{i+1}_loss{loss:.6f}.npz"
            np.savez(os.path.join(folder, fname),
                     unitary=np.asarray(U),
                     thetas=np.asarray(thetas),
                     phis=np.asarray(phis))
            saved.append(fname)
            
        messagebox.showinfo("Export Successful", f"Saved {top_n} unitaries:\n" + "\n".join(saved))

    def _on_error_change(self, event=None):
        """Updates the beamsplitter error model in the engine."""
        mode = self.error_mode.get()
        if mode == "Ideal (No Error)":
            self.engine.set_bs_error(0.0)
        elif mode == "Basic Loss (15%)":
            self.engine.set_bs_error(0.15)
        elif mode == "Calibration Data":
            import os
            json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'node-isolation', '8-mode-autocal-20260209.json'))
            if os.path.exists(json_path):
                self.engine.load_calibration_errors(json_path, default_e=0.0)
            else:
                print("Warning: Calibration JSON not found.")
                self.engine.set_bs_error(0.0)
        self._update_simulation()

    def _change_input(self, idx, delta):
        """Updates the photon/power input count for a specific spatial mode."""
        new_val = max(0, self.input_vars[idx] + delta)
        self.input_vars[idx] = new_val
        self.input_labels[idx].config(text=str(new_val))
        if not self.pr_active:
            self._update_simulation()

    def _set_modes(self, n):
        """Rebuilds the entire simulation and UI for a different number of spatial modes."""
        if n == self.n_modes:
            return
            
        self.main_container.destroy()
        
        self.n_modes = n
        self.root.title(f"Interferometer Simulator ({n} Modes)")
        self.engine = Engine(n_modes=n)
        self.selected_mzi = None
        self.input_vars = [0] * n
        self.pr_active = False
        self.target_vars = [tk.DoubleVar(value=0.0) for _ in range(n)]
        
        self.phases = {}
        for mid in self.engine.mzi_ids:
            self.phases[mid] = {'theta': float(jnp.pi), 'phi': 0.0}
        
        self._create_layout()
        self._on_error_change() # Reapply selected error mode
        self.root.update()
        self._draw_mesh()
        self._update_simulation()

    def _get_phase_arrays(self):
        """Extracts thetas and phis as JAX arrays for the pure engine functions."""
        thetas = jnp.array([self.phases[mid]['theta'] for mid in self.engine.mzi_ids], dtype=jnp.float64)
        phis = jnp.array([self.phases[mid]['phi'] for mid in self.engine.mzi_ids], dtype=jnp.float64)
        return thetas, phis

    def _draw_mesh(self, event=None):
        """Renders the photonic mesh and classical power flow on the canvas."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10: return 

        # margins and grid sizing
        mx, my = 60, 60
        eff_w = w - 2*mx
        eff_h = h - 2*my
        
        col_width = eff_w / self.n_modes
        row_height = eff_h / self.n_modes
        
        self.mzi_coords = {}
        
        # calculate power flow
        input_vec = self.input_vars
        total_p = sum(input_vec)
        thetas, phis = self._get_phase_arrays()
        
        is_coherent = (self.sim_mode.get() == "quantum")
        powers_jax = self.engine.get_classical_flow(thetas, phis, input_vec, coherent=is_coherent)
        powers = [np.asarray(p) for p in powers_jax]
        max_p = np.max(powers) if np.max(powers) > 0 else 1.0

        def get_color(intensity):
            """Maps light intensity to a color for waveguide visualization."""
            if intensity < 0.001: return self.colors['waveguide_off']
            norm = min(1.0, intensity / max_p)
            
            # Apply a baseline threshold to prevent black highlight for low intensity
            norm = 0.4 + 0.6 * norm
            
            r = int(min(255, norm * 255))
            g = int(min(255, norm * 200)) 
            b = int(min(50, norm * 50))
            return f"#{r:02x}{g:02x}{b:02x}"
        
        for i in range(self.n_modes):
            y = my + i * row_height + row_height / 2
            self.canvas.create_text(mx - 40, y, text=f"{i+1}", fill="#888", font=("Arial", 10))  # left labels 
            self.canvas.create_text(w - mx + 40, y, text=f"{i+1}", fill="#888", font=("Arial", 10)) # right labels

        # draw initial input lines (before first column)
        initial_powers = powers[0]
        for mode in range(self.n_modes):
            y_mode = my + mode * row_height + row_height/2
            intensity = initial_powers[mode]
            color = get_color(intensity)
            width = 2 + (intensity/max_p)*3 if intensity > 0 else 2
            
            self.canvas.create_line(mx - 20, y_mode, mx, y_mode, 
                                    fill=color, width=width, capstyle=tk.ROUND)
        
        for col_idx in range(self.n_modes): 
            x_start = mx + col_idx * col_width
            x_end = mx + (col_idx + 1) * col_width
            
            # current intensities at the start of this column
            current_intensities = powers[col_idx] if col_idx < len(powers) else np.zeros(self.n_modes)
            
            mzi_map = {} # mode -> mzi_id
            if col_idx < len(self.engine.layout):
                for mzi in self.engine.layout[col_idx]:
                    mzi_map[mzi['mode_top']] = mzi
                    mzi_map[mzi['mode_top']+1] = mzi
            
            # draw lines
            processed_modes = set()
            
            for mode in range(self.n_modes):
                if mode in processed_modes: continue
                
                y_mode = my + mode * row_height + row_height/2
                intensity = current_intensities[mode]
                color = get_color(intensity)
                width = 2 + (intensity/max_p)*3 if intensity > 0 else 2
                
                if mode in mzi_map:
                    # this mode is part of an MZI
                    mzi = mzi_map[mode]
                    mzi_id = mzi['id']
                    top = mzi['mode_top']
                    
                    # if we are at the top mode of the MZI, draw the MZI structure
                    if mode == top:
                        bot = top + 1
                        y_top = my + top * row_height + row_height/2
                        y_bot = my + bot * row_height + row_height/2
                        
                        # store coordinates for clicking
                        mid_x = (x_start + x_end) / 2
                        mid_y = (y_top + y_bot) / 2
                        self.mzi_coords[mzi_id] = (mid_x, mid_y, 25)
                        
                        # intensities at start and end of this column
                        in_intensities = current_intensities
                        out_intensities = powers[col_idx + 1] if col_idx + 1 < len(powers) else in_intensities
                        
                        # incoming segments
                        int_top_in = in_intensities[top]
                        int_bot_in = in_intensities[bot]
                        col_top_in = get_color(int_top_in)
                        col_bot_in = get_color(int_bot_in)
                        w_top_in = 2 + (int_top_in/max_p)*3 if int_top_in > 0 else 2
                        w_bot_in = 2 + (int_bot_in/max_p)*3 if int_bot_in > 0 else 2

                        # outgoing segments
                        int_top_out = out_intensities[top]
                        int_bot_out = out_intensities[bot]
                        col_top_out = get_color(int_top_out)
                        col_bot_out = get_color(int_bot_out)
                        w_top_out = 2 + (int_top_out/max_p)*3 if int_top_out > 0 else 2
                        w_bot_out = 2 + (int_bot_out/max_p)*3 if int_bot_out > 0 else 2
                        
                        # phase shifter position (on top arm, before center)
                        ps_x = x_start + (mid_x - x_start) * 0.5
                        y_ps_linear = y_top + (mid_y - y_top) * 0.5                       

                        # draw the four segments
                        self.canvas.create_line(x_start, y_top, mid_x, mid_y, fill=col_top_in, width=w_top_in, capstyle=tk.ROUND)
                        self.canvas.create_line(mid_x, mid_y, x_end, y_top, fill=col_top_out, width=w_top_out, capstyle=tk.ROUND)
                        self.canvas.create_line(x_start, y_bot, mid_x, mid_y, fill=col_bot_in, width=w_bot_in, capstyle=tk.ROUND)
                        self.canvas.create_line(mid_x, mid_y, x_end, y_bot, fill=col_bot_out, width=w_bot_out, capstyle=tk.ROUND)
                        
                        # MZI selection visuals
                        is_sel = (self.selected_mzi == mzi_id)
                        accent_col = self.colors['accent'] if is_sel else "#555"
                        
                        # get current phases
                        curr_theta = self.phases[mzi_id]['theta']
                        curr_phi = self.phases[mzi_id]['phi']

                        # draw selection ring around the crossing point
                        if is_sel:
                            r_sel = 15
                            self.canvas.create_oval(mid_x - r_sel, mid_y - r_sel, 
                                                    mid_x + r_sel, mid_y + r_sel,
                                                    outline=self.colors['accent'], width=2)
                        
                        # label for theta
                        self.canvas.create_text(mid_x + 20, mid_y + 4, 
                                                text=f"θ:{curr_theta/jnp.pi:.2f}π", 
                                                fill="#aaa", font=("Arial", 7), anchor="w", angle=30)

                        # draw phase shifter on top arm
                        r_ps = 5
                        self.canvas.create_oval(ps_x - r_ps, y_ps_linear - r_ps, 
                                                ps_x + r_ps, y_ps_linear + r_ps,
                                                fill="#222", outline=accent_col, width=1)
                        
                        # label for phi
                        self.canvas.create_text(ps_x + 8, y_ps_linear - 12, 
                                                text=f"φ:{curr_phi/jnp.pi:.2f}π", 
                                                fill="#aaa", font=("Arial", 7), anchor="w", angle=30)
                        
                        # small ID label below
                        self.canvas.create_text(mid_x, mid_y + 20, text=mzi_id, fill="#666", font=("Arial", 8))
                        
                        processed_modes.add(top)
                        processed_modes.add(bot)
                        
                else:
                    # straight waveguide
                    self.canvas.create_line(x_start, y_mode, x_end, y_mode, 
                                          fill=color, width=width, capstyle=tk.ROUND)
        
        # draw final output lines (after last column)
        last_col_x = mx + self.n_modes * col_width
        final_powers = powers[-1]
        for mode in range(self.n_modes):
            y_mode = my + mode * row_height + row_height/2
            intensity = final_powers[mode]
            color = get_color(intensity)
            width = 2 + (intensity/max_p)*3 if intensity > 0 else 2
            
            self.canvas.create_line(last_col_x, y_mode, w - mx + 20, y_mode, 
                                    fill=color, width=width, capstyle=tk.ROUND)

    def _on_canvas_click(self, event):
        """Detects MZI selection based on click coordinates on the canvas."""
        x, y = event.x, event.y
        closest_dist = 9999
        closest_id = None
        
        for mid, (mx, my, r) in self.mzi_coords.items():
            dist = math.sqrt((x-mx)**2 + (y-my)**2)
            if dist < r + 10: # hit radius
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = mid
        
        if closest_id:
            self.selected_mzi = closest_id
            self.lbl_selected.config(text=f"Editing MZI: {closest_id}", foreground=self.colors['accent'])
            
            p = self.phases[closest_id]
            self.theta_var.set(p['theta'] / float(jnp.pi))
            self.phi_var.set(p['phi'] / float(jnp.pi))
            self._draw_mesh() # redraw to show selection highlight

    def _on_param_change(self):
        """Handles phase parameter updates and refreshes the simulation."""
        if self.selected_mzi:
            self.phases[self.selected_mzi]['theta'] = self.theta_var.get() * float(jnp.pi)
            self.phases[self.selected_mzi]['phi'] = self.phi_var.get() * float(jnp.pi)
            self._update_simulation()

    def _set_preset(self, t, p):
        """Applies predefined phase settings to the selected MZI."""
        self.theta_var.set(t)
        self.phi_var.set(p)
        self._on_param_change()

    def _demo_hom(self):
        """Sets up the classic Hong-Ou-Mandel interference test."""
        self._set_sim_mode("quantum")
        # Clear all inputs and MZIs
        self._demo_clear(update=False)
        
        # Inject two indistinguishable photons in ports 0 and 1
        self.input_vars[0] = 1
        self.input_vars[1] = 1
        self.input_labels[0].config(text="1")
        self.input_labels[1].config(text="1")
        
        # Set first MZI (A1) to a 50:50 beam splitter
        first_mzi = self.engine.layout[0][0]['id']
        self.phases[first_mzi]['theta'] = float(jnp.pi / 2)
        
        if self.selected_mzi == first_mzi:
            self.theta_var.set(0.5)
            
        self._update_simulation()

    def _demo_random(self):
        """Sets up a random boson sampling experiment with 3 photons."""
        self._set_sim_mode("quantum")
        import random
        # Clear
        self._demo_clear(update=False)
        
        # Add 3 random photons
        for _ in range(3):
            idx = random.randint(0, self.n_modes - 1)
            self.input_vars[idx] += 1
            self.input_labels[idx].config(text=str(self.input_vars[idx]))
            
        # Randomize all phases
        for mid in self.engine.mzi_ids:
            t_pi = random.uniform(0, 1.0)
            p_pi = random.uniform(0, 2.0)
            self.phases[mid]['theta'] = t_pi * float(jnp.pi)
            self.phases[mid]['phi'] = p_pi * float(jnp.pi)
            
            if self.selected_mzi == mid:
                self.theta_var.set(t_pi)
                self.phi_var.set(p_pi)
                
        self._update_simulation()

    def _demo_unitary_decomposition(self):
        """Generates a random unitary, decomposes it via pnn.py, and sets the MZI phases."""
        dim = self.n_modes
        Z = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        Q, R = np.linalg.qr(Z)
        d = np.diagonal(R)
        ph = d / np.abs(d)
        U_rand = Q @ np.diag(ph)
        
        self._apply_unitary_decomposition(U_rand)

    def _demo_switching_decomposition(self):
        """Generates a random permutation (switching) matrix and applies it to the mesh."""
        dim = self.n_modes
        P = np.zeros((dim, dim), dtype=np.complex128)
        cols = np.random.permutation(dim)
        for i in range(dim):
            P[i, cols[i]] = 1.0
            
        self._apply_unitary_decomposition(P)

    def _import_unitary_decomposition(self):
        """Opens a file dialog to import a .npy or .npz unitary matrix and visualizes it."""
        from tkinter import filedialog
        import os
        filepath = filedialog.askopenfilename(
            title="Select Unitary File",
            filetypes=[("Numpy files", "*.npy *.npz"), ("All files", "*.*")]
        )
        if not filepath:
            return
            
        try:
            data = np.load(filepath, allow_pickle=False)
            
            # Check if this is a .npz with embedded phases (exported from phase retrieval)
            if isinstance(data, np.lib.npyio.NpzFile) and 'thetas' in data and 'phis' in data:
                thetas = data['thetas']
                phis = data['phis']
                U = data.get('unitary', None)
                
                # Resize mesh if needed
                n_mzis = len(self.engine.mzi_ids)
                if len(thetas) != n_mzis:
                    # Try to infer mode count from unitary shape
                    if U is not None and len(U.shape) == 2:
                        self._set_modes(U.shape[0])
                
                # Apply phases directly — no Clements decomposition
                for i, mid in enumerate(self.engine.mzi_ids):
                    self.phases[mid]['theta'] = float(thetas[i])
                    self.phases[mid]['phi'] = float(phis[i])
                    if self.selected_mzi == mid:
                        self.theta_var.set(float(thetas[i]) / float(np.pi))
                        self.phi_var.set(float(phis[i]) / float(np.pi))
                
                self.lbl_unitary.config(text=f"Loaded phases: {os.path.basename(filepath)}")
                self._update_simulation()
            else:
                # Plain unitary matrix — use Clements decomposition
                if isinstance(data, np.lib.npyio.NpzFile):
                    U_loaded = data[data.files[0]]
                else:
                    U_loaded = data
                    
                if len(U_loaded.shape) != 2 or U_loaded.shape[0] != U_loaded.shape[1]:
                    raise ValueError("Array must be a 2D square matrix.")
                    
                if U_loaded.shape[0] != self.n_modes:
                    print(f"Switching mesh to {U_loaded.shape[0]} modes to match loaded unitary.")
                    self._set_modes(U_loaded.shape[0])
                    
                self._apply_unitary_decomposition(U_loaded)
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Import Error", f"Failed to load unitary:\n{e}")

    def _import_unitary_folder(self):
        """Opens a file dialog to import a folder containing unitary matrices."""
        from tkinter import filedialog
        import os
        folderpath = filedialog.askdirectory(title="Select Folder containing Unitaries")
        if not folderpath:
            return
            
        try:
            files = sorted([os.path.join(folderpath, f) for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f))])
            unitaries = []
            valid_files = []
            for f in files:
                try:
                    if f.endswith('.npy'):
                        U = np.load(f)
                    elif f.endswith('.npz'):
                        U_loaded = np.load(f)
                        U = U_loaded[U_loaded.files[0]]
                    elif f.endswith('.txt') or f.endswith('.csv'):
                        U = np.loadtxt(f, dtype=np.complex128)
                    else:
                        try:
                            U = np.load(f)
                        except:
                            U = np.loadtxt(f, dtype=np.complex128)
                            
                    if len(U.shape) == 2 and U.shape[0] == U.shape[1]:
                        unitaries.append(U)
                        valid_files.append(os.path.basename(f))
                except:
                    continue
                    
            if not unitaries:
                from tkinter import messagebox
                messagebox.showerror("Import Error", "No valid unitary matrices found in folder.")
                return
                
            self.loaded_unitaries = unitaries
            self.unitary_files = valid_files
            self.current_unitary_idx = 0
            
            if unitaries[0].shape[0] != self.n_modes:
                self._set_modes(unitaries[0].shape[0])
                
            self._load_unitary_from_list()
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Import Error", f"Failed to load folder:\n{e}")

    def _load_unitary_from_list(self):
        """Loads and visualizes the currently selected unitary from the loaded list."""
        if not self.loaded_unitaries:
            return
        # In phase retrieval mode, apply phases directly (skip Clements)
        if self.pr_active and hasattr(self, '_pr_results') and self._pr_results:
            self._apply_pr_result(self.current_unitary_idx)
        else:
            U = self.loaded_unitaries[self.current_unitary_idx]
            fname = self.unitary_files[self.current_unitary_idx]
            self.lbl_unitary.config(text=f"Loaded: {fname} ({self.current_unitary_idx + 1} / {len(self.loaded_unitaries)})")
            self._apply_unitary_decomposition(U, keep_inputs=True, keep_mode=True)

    def _prev_unitary(self):
        """Cycles to the previous unitary in the loaded list."""
        if self.loaded_unitaries:
            self.current_unitary_idx = (self.current_unitary_idx - 1) % len(self.loaded_unitaries)
            self._load_unitary_from_list()
            
    def _next_unitary(self):
        """Cycles to the next unitary in the loaded list."""
        if self.loaded_unitaries:
            self.current_unitary_idx = (self.current_unitary_idx + 1) % len(self.loaded_unitaries)
            self._load_unitary_from_list()

    def _apply_unitary_decomposition(self, U_target, keep_inputs=False, keep_mode=False):
        """Decomposes a given unitary matrix and assigns the phases to the mesh."""
        if keep_inputs:
            saved_inputs = list(self.input_vars)
            
        if not keep_mode:
            self._set_sim_mode("classical")
            
        self._demo_clear(update=False)
        
        try:
            import sys
            import os
            pnn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if pnn_dir not in sys.path:
                sys.path.append(pnn_dir)
            from decompose.pnn import decompose_clements
            
            phis, thetas, alphas = decompose_clements(U_target, block='mzi')
            
            # Map to engine layout
            # pnn.py 'theta' is half of engine's 'theta'
            for col_idx, col in enumerate(self.engine.layout):
                p = col_idx // 2
                for mzi in col:
                    q = mzi['mode_top']
                    mid = mzi['id']
                    
                    theta_val = float(np.mod(2 * thetas[q, p], 2 * np.pi))
                    phi_val = float(np.mod(phis[q, p], 2 * np.pi))
                    
                    self.phases[mid]['theta'] = theta_val
                    self.phases[mid]['phi'] = phi_val
                    
                    if self.selected_mzi == mid:
                        self.theta_var.set(theta_val / float(jnp.pi))
                        self.phi_var.set(phi_val / float(jnp.pi))
                        
            if keep_inputs:
                for i in range(self.n_modes):
                    self.input_vars[i] = saved_inputs[i]
                    self.input_labels[i].config(text=str(saved_inputs[i]))
            else:
                # Set classical input to port 1 so we can see the routing
                self.input_vars[0] = 1
                self.input_labels[0].config(text="1")
            
            self._update_simulation()
            
        except Exception as e:
            print(f"Decomposition failed: {e}")

    def _demo_clear(self, update=True):
        """Resets the mesh to Identity (all Bar state) and clears photons."""
        for i in range(self.n_modes):
            self.input_vars[i] = 0
            self.input_labels[i].config(text="0")
            
        for mid in self.engine.mzi_ids:
            self.phases[mid]['theta'] = float(jnp.pi)
            self.phases[mid]['phi'] = 0.0
            
            if self.selected_mzi == mid:
                self.theta_var.set(1.0)
                self.phi_var.set(0.0)
                
        if update:
            self._update_simulation()

    def _update_simulation(self):
        """Updates the simulation results and refreshes all visualizations."""
        self._draw_mesh()
        self.root.update_idletasks() # Force UI to render the mesh immediately

        input_vec = self.input_vars
        
        # clear plot and setup base style
        self.ax.clear()
        self.ax.set_facecolor(self.colors['bg'])
        
        # remove spines for clean look (default state)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(axis='both', which='both', length=0) # hide ticks by default
            
        if self.sim_mode.get() == "quantum":
            thetas, phis = self._get_phase_arrays()
            probs_jax, basis = self.engine.propagate_fock(thetas, phis, input_vec)
            
            # Convert to dict and filter small probabilities to maintain UI performance
            results = {}
            if len(probs_jax) > 0:
                probs_np = np.asarray(probs_jax)
                for i, out_state in enumerate(basis):
                    p = float(probs_np[i])
                    if p > 0.0001:
                        results[out_state] = p
            
            # sort and take top 12 results
            sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
            top_res = sorted_res[:12] 
            
            labels = [f"|{''.join(map(str, s))}>" for s, p in top_res]
            probs = [p for s, p in top_res]
            title = "State probability"
        else:
            # Classical mode
            # Incoherent power tracking for independent classical sources
            thetas, phis = self._get_phase_arrays()
            U = self.engine.compute_full_unitary(thetas, phis)
            power_trans = np.abs(U) ** 2
            input_p = np.array(input_vec, dtype=np.float64)
            out_powers = power_trans @ input_p
            self.phases_out = np.zeros(self.n_modes)
            
            labels = [f"Port {i+1}" for i in range(self.n_modes)]
            probs = out_powers.tolist()
            title = "Output Optical Power & Phase"

        y_pos = np.arange(len(labels))

        if not probs or sum(probs) == 0:
            self.ax.axis('off')
            self.canvas_plot.draw()
            return
        
        self.ax.hlines(y=y_pos, xmin=0, xmax=probs, color=self.colors['accent'], alpha=0.5, linewidth=2)
        self.ax.plot(probs, y_pos, 'o', color=self.colors['accent'], markersize=8, markeredgecolor='white', markeredgewidth=1.5)

        max_p = max(probs)
        for i, (p, label) in enumerate(zip(probs, labels)):
            if self.sim_mode.get() == "quantum":
                # percentage at the end
                self.ax.text(p + (max_p * 0.07), i, f"{p*100:.1f}%", va='center', color='white', fontsize=9, fontweight='bold')
                # label above the bar
                self.ax.text(0, i - 0.15, label, ha='left', va='bottom', color='white', fontsize=10, family='monospace')
            else:
                self.ax.text(p + (max_p * 0.07), i, f"{p:.5f}", va='center', color='white', fontsize=9, fontweight='bold')
                phase_val_pi = self.phases_out[i] / np.pi
                self.ax.text(0, i - 0.15, f"{label} (φ: {phase_val_pi:.2f}π)", ha='left', va='bottom', color='white', fontsize=10, family='monospace')

        # final styling
        self.ax.axis('off')
        self.ax.invert_yaxis()  
        self.ax.set_ylim(max(11.5, len(labels) - 0.5), -0.5) 
        self.ax.set_title(title, color="white", pad=10, fontsize=12, fontweight='bold')        
        self.ax.set_xlim(0, max_p * 1.35 if max_p > 0 else 1) 
        self.fig.tight_layout(pad=2)
        self.canvas_plot.draw()