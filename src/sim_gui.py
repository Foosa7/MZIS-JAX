import tkinter as tk
from tkinter import ttk

def build_sim_controls(gui, parent):
    """Builds the control panels for the Simulator mode."""
    # Hardware Errors
    ttk.Label(parent, text="Hardware Imperfections", style="Header.TLabel").pack(anchor="w", pady=(15, 5))
    hw_frame = ttk.Frame(parent, style="Panel.TFrame")
    hw_frame.pack(fill=tk.X, pady=(0, 15))
    
    gui.error_mode = tk.StringVar(value="Ideal (No Error)")
    gui.error_dropdown = tk.OptionMenu(hw_frame, gui.error_mode, 
                                        "Ideal (No Error)", "Beamsplitter Errors (15%)", "Calibration Data",
                                        command=gui._on_error_change)
    gui.error_dropdown.config(bg="#333", fg="white", activebackground="#444", activeforeground="white",
                                bd=0, highlightthickness=0, font=("Arial", 10, "bold"))
    gui.error_dropdown["menu"].config(bg="#333", fg="white", activebackground=gui.colors['accent'], activeforeground="black", font=("Arial", 10))
    gui.error_dropdown.pack(fill=tk.X, pady=5)

    ttk.Label(parent, text="Light Inputs", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
    
    input_grid = ttk.Frame(parent, style="Panel.TFrame")
    input_grid.pack(fill=tk.X)
    
    gui.input_labels = []
    for i in range(gui.n_modes):
        row = ttk.Frame(input_grid, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        
        ttk.Label(row, text=f"Port {i+1}", style="Panel.TLabel", width=8).pack(side=tk.LEFT)
        
        # minus button
        btn_minus = tk.Button(row, text="-", width=3, bg="#333", fg="white", bd=0,
                                activebackground="#444", activeforeground="white",
                                command=lambda idx=i: gui._change_input(idx, -1))
        btn_minus.pack(side=tk.LEFT, padx=2)
        
        # value label
        lbl = tk.Label(row, text="0", width=4, bg="#1e1e1e", fg=gui.colors['accent'], font=("Arial", 11, "bold"))
        lbl.pack(side=tk.LEFT, padx=2)
        gui.input_labels.append(lbl)
        
        # plus button
        btn_plus = tk.Button(row, text="+", width=3, bg="#333", fg="white", bd=0,
                                activebackground="#444", activeforeground="white",
                                command=lambda idx=i: gui._change_input(idx, 1))
        btn_plus.pack(side=tk.LEFT, padx=2)

    ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=20)

    gui.dynamic_frame = ttk.Frame(parent, style="Panel.TFrame")
    gui.dynamic_frame.pack(fill=tk.X)

    gui.target_frame = ttk.Frame(gui.dynamic_frame, style="Panel.TFrame")
    
    ttk.Label(gui.target_frame, text="Target Outputs (Phase Retrieval)", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
    for i in range(gui.n_modes):
        f = ttk.Frame(gui.target_frame, style="Panel.TFrame")
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=f"Port {i+1}", style="Panel.TLabel", width=8).pack(side=tk.LEFT)
        scale = tk.Scale(f, from_=0, to=1.0, variable=gui.target_vars[i], orient=tk.HORIZONTAL, 
                            bg=gui.colors['panel'], fg="white", troughcolor="#111",
                            activebackground=gui.colors['accent'], highlightthickness=0, bd=0,
                            resolution=0.01)
        scale.pack(fill=tk.X, side=tk.LEFT, expand=True)
    
    btn_compute = tk.Button(gui.target_frame, text="Compute Routing", bg=gui.colors['accent'], fg="black",
                            font=("Arial", 11, "bold"), bd=0, activebackground="#00cc99",
                            command=gui._on_target_change)
    btn_compute.pack(fill=tk.X, pady=(10, 5), ipady=6)
        
    export_frame = ttk.Frame(gui.target_frame, style="Panel.TFrame")
    export_frame.pack(fill=tk.X, pady=(0, 5))
    btn_export = ttk.Button(export_frame, text="Export Current Unitary", command=gui._export_current_unitary)
    btn_export.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
    btn_export_top3 = ttk.Button(export_frame, text="Export Top 3", command=gui._export_top3_unitaries)
    btn_export_top3.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
    
    btn_reset_pr = ttk.Button(gui.target_frame, text="Reset to Identity", command=gui._demo_clear)
    btn_reset_pr.pack(fill=tk.X, pady=(0, 5))

    gui.mzi_frame = ttk.Frame(gui.dynamic_frame, style="Panel.TFrame")
    gui.mzi_frame.pack(fill=tk.X)
    gui.target_frame.pack_forget()
    
    ttk.Label(gui.mzi_frame, text="MZI Config", style="Header.TLabel").pack(anchor="w")
    gui.lbl_selected = ttk.Label(gui.mzi_frame, text="Select an MZI on the mesh", 
                                    foreground="#888", style="Panel.TLabel")
    gui.lbl_selected.pack(pady=(5, 15), anchor="w")

    gui.theta_var = tk.DoubleVar()
    gui.phi_var = tk.DoubleVar()
    
    def create_slider(parent_widget, label, var, r_max):
        f = ttk.Frame(parent_widget, style="Panel.TFrame")
        f.pack(fill=tk.X, pady=5)
        ttk.Label(f, text=label, style="Panel.TLabel").pack(anchor="w")
        scale = tk.Scale(f, from_=0, to=r_max, variable=var, orient=tk.HORIZONTAL, 
                            bg=gui.colors['panel'], fg="white", troughcolor="#111",
                            activebackground=gui.colors['accent'], highlightthickness=0, bd=0,
                            resolution=0.01, command=lambda x: gui._on_param_change())
        scale.pack(fill=tk.X)
        
    create_slider(gui.mzi_frame, "Internal phase (θ) - Splitting (xπ)", gui.theta_var, 2.0)
    create_slider(gui.mzi_frame, "External phase (φ) - Phase Shift (xπ)", gui.phi_var, 2.0)

    # preset buttons
    ttk.Label(gui.mzi_frame, text="Quick Presets", style="Panel.TLabel").pack(anchor="w", pady=(15, 5))
    btn_frame = ttk.Frame(gui.mzi_frame, style="Panel.TFrame")
    btn_frame.pack(fill=tk.X)
    
    def mk_btn(txt, t, p):
        b = ttk.Button(btn_frame, text=txt, command=lambda: gui._set_preset(t, p))
        b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
    mk_btn("Bar", 1.0, 0.0)
    mk_btn("50:50", 0.5, 0.0)
    mk_btn("Cross", 0.0, 0.0)

    # Demos
    ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=20)
    
    gui.demo_frame = ttk.Frame(parent, style="Panel.TFrame")
    gui.demo_frame.pack(fill=tk.X)
    ttk.Label(gui.demo_frame, text="Demos", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
    
    btn_hom = ttk.Button(gui.demo_frame, text="HOM Dip (2 Photons)", command=gui._demo_hom)
    btn_hom.pack(fill=tk.X, pady=2)
    
    btn_rand = ttk.Button(gui.demo_frame, text="Random Boson Sampling", command=gui._demo_random)
    btn_rand.pack(fill=tk.X, pady=2)
    
    decomp_frame = ttk.Frame(gui.demo_frame, style="Panel.TFrame")
    decomp_frame.pack(fill=tk.X, pady=2)
    
    btn_haar = ttk.Button(decomp_frame, text="Haar Random", command=gui._demo_unitary_decomposition)
    btn_haar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
    
    btn_switch = ttk.Button(decomp_frame, text="Switching Matrix", command=gui._demo_switching_decomposition)
    btn_switch.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
    
    import_export_frame = ttk.Frame(gui.demo_frame, style="Panel.TFrame")
    import_export_frame.pack(fill=tk.X, pady=2)
    
    btn_import = ttk.Button(import_export_frame, text="Import Unitary", command=gui._import_unitary_decomposition)
    btn_import.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
    
    btn_export = ttk.Button(import_export_frame, text="Export Clements", command=gui._export_clements_decomposition)
    btn_export.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 2))

    btn_opt = ttk.Button(import_export_frame, text="🪄", command=gui._optimize_unitary)
    btn_opt.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
    
    btn_import_folder = ttk.Button(gui.demo_frame, text="Import Unitary Folder", command=gui._import_unitary_folder)
    btn_import_folder.pack(fill=tk.X, pady=2)
    
    btn_clear = ttk.Button(gui.demo_frame, text="Reset to Identity", command=gui._demo_clear)
    btn_clear.pack(fill=tk.X, pady=2)
