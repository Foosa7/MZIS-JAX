import tkinter as tk
from tkinter import ttk
import numpy as np
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .engine import Engine

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

        self.engine = Engine(n_modes=self.n_modes, bs_error=0.15)
        self.selected_mzi = None
        self.input_vars = [0] * n_modes
        self.sim_mode = tk.StringVar(value="quantum")
        
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
        
        btn_8 = ttk.Button(size_frame, text="N=8", command=lambda: self._set_modes(8))
        btn_8.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        btn_12 = ttk.Button(size_frame, text="N=12", command=lambda: self._set_modes(12))
        btn_12.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        btn_16 = ttk.Button(size_frame, text="N=16", command=lambda: self._set_modes(16))
        btn_16.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Label(pad_frame, text="Simulation Mode", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        mode_frame = ttk.Frame(pad_frame, style="Panel.TFrame")
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.btn_q = tk.Button(mode_frame, text="Quantum", bg=self.colors['accent'], fg="black", bd=0, 
                               font=("Arial", 10, "bold"), command=lambda: self._set_sim_mode("quantum"))
        self.btn_q.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        self.btn_c = tk.Button(mode_frame, text="Classical", bg="#333", fg="white", bd=0, 
                               font=("Arial", 10, "bold"), command=lambda: self._set_sim_mode("classical"))
        self.btn_c.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        
        # Initialize button colors
        if self.sim_mode.get() == "classical":
            self.btn_q.config(bg="#333", fg="white")
            self.btn_c.config(bg=self.colors['accent'], fg="black")

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

        self.mzi_frame = ttk.Frame(pad_frame, style="Panel.TFrame")
        self.mzi_frame.pack(fill=tk.X)
        
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
        
        btn_clear = ttk.Button(self.demo_frame, text="Reset to Identity", command=self._demo_clear)
        btn_clear.pack(fill=tk.X, pady=2)

        center_area = ttk.Frame(self.main_container, style="TFrame")
        center_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # mesh canvas 
        self.canvas = tk.Canvas(center_area, bg=self.colors['bg'], highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._draw_mesh)

        # graph 
        self.plot_frame = ttk.Frame(center_area, width=350, style="TFrame")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        self.plot_frame.pack_propagate(False) # Fix width
        
        self.fig = plt.Figure(figsize=(4, 8), facecolor=self.colors['bg'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['bg'])
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _set_sim_mode(self, mode):
        """Sets the simulation mode and updates UI."""
        self.sim_mode.set(mode)
        if mode == "quantum":
            self.btn_q.config(bg=self.colors['accent'], fg="black")
            self.btn_c.config(bg="#333", fg="white")
        else:
            self.btn_q.config(bg="#333", fg="white")
            self.btn_c.config(bg=self.colors['accent'], fg="black")
        self._on_sim_mode_change()

    def _on_sim_mode_change(self):
        """Resets inputs and switches simulation mode."""
        for i in range(self.n_modes):
            self.input_vars[i] = 0
            self.input_labels[i].config(text="0")
        self._update_simulation()

    def _change_input(self, idx, delta):
        """Updates the photon/power input count for a specific spatial mode."""
        new_val = max(0, self.input_vars[idx] + delta)
        self.input_vars[idx] = new_val
        self.input_labels[idx].config(text=str(new_val))
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
        
        self.phases = {}
        for mid in self.engine.mzi_ids:
            self.phases[mid] = {'theta': float(jnp.pi), 'phi': 0.0}
        
        self._create_layout()
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
        powers_jax = self.engine.get_classical_flow(thetas, phis, input_vec, coherent=True)
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
        filepath = filedialog.askopenfilename(
            title="Select Unitary File",
            filetypes=[("Numpy arrays", "*.npy *.npz"), ("All files", "*.*")]
        )
        if not filepath:
            return
            
        try:
            U_loaded = np.load(filepath)
            if filepath.endswith('.npz'):
                U_loaded = U_loaded[U_loaded.files[0]]
                
            if len(U_loaded.shape) != 2 or U_loaded.shape[0] != U_loaded.shape[1]:
                raise ValueError("Array must be a 2D square matrix.")
                
            if U_loaded.shape[0] != self.n_modes:
                print(f"Switching mesh to {U_loaded.shape[0]} modes to match loaded unitary.")
                self._set_modes(U_loaded.shape[0])
                
            self._apply_unitary_decomposition(U_loaded)
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Import Error", f"Failed to load unitary:\n{e}")

    def _apply_unitary_decomposition(self, U_target):
        """Decomposes a given unitary matrix and assigns the phases to the mesh."""
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
            # Coherent powers and phases
            thetas, phis = self._get_phase_arrays()
            U = self.engine.compute_full_unitary(thetas, phis)
            amps_in = np.sqrt(input_vec)
            amps_out = np.asarray(U @ amps_in)
            out_powers = np.abs(amps_out) ** 2
            self.phases_out = np.mod(np.angle(amps_out), 2*np.pi)
            
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
                self.ax.text(p + (max_p * 0.07), i, f"{p:.2f}", va='center', color='white', fontsize=9, fontweight='bold')
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