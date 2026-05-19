import tkinter as tk
from tkinter import ttk

def build_hw_controls(gui, parent):
    """Builds the control panels for Hardware/Live mode."""
    ttk.Label(parent, text="Hardware Job Queue", style="Header.TLabel").pack(anchor="w", pady=(15, 5))
    
    btn_import = tk.Button(parent, text="Import Preprocessed Batch", bg=gui.colors['accent'], fg="black", bd=0,
                           font=("Arial", 11, "bold"), activebackground="#00cc99",
                           command=gui._import_hw_batch)
    btn_import.pack(fill=tk.X, pady=(10, 20), ipady=8)
    
    # Telemetry section
    ttk.Label(parent, text="Live Telemetry", style="Header.TLabel").pack(anchor="w", pady=(10, 5))
    
    telemetry_frame = ttk.Frame(parent, style="Panel.TFrame")
    telemetry_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(telemetry_frame, text="Status:", style="Panel.TLabel", width=12).grid(row=0, column=0, sticky="w", pady=2)
    gui.lbl_hw_status = ttk.Label(telemetry_frame, text="Idle", foreground="#888", style="Panel.TLabel")
    gui.lbl_hw_status.grid(row=0, column=1, sticky="w", pady=2)
    
    ttk.Label(telemetry_frame, text="Target:", style="Panel.TLabel", width=12).grid(row=1, column=0, sticky="w", pady=2)
    gui.lbl_hw_target = ttk.Label(telemetry_frame, text="-", foreground="#888", style="Panel.TLabel")
    gui.lbl_hw_target.grid(row=1, column=1, sticky="w", pady=2)
    
    ttk.Label(telemetry_frame, text="Iteration:", style="Panel.TLabel", width=12).grid(row=2, column=0, sticky="w", pady=2)
    gui.lbl_hw_iter = ttk.Label(telemetry_frame, text="-", foreground="#888", style="Panel.TLabel")
    gui.lbl_hw_iter.grid(row=2, column=1, sticky="w", pady=2)
    
    ttk.Label(telemetry_frame, text="Loss:", style="Panel.TLabel", width=12).grid(row=3, column=0, sticky="w", pady=2)
    gui.lbl_hw_loss = ttk.Label(telemetry_frame, text="-", foreground="#888", style="Panel.TLabel")
    gui.lbl_hw_loss.grid(row=3, column=1, sticky="w", pady=2)
