import tkinter as tk
from src import GUI

if __name__ == "__main__":
    root = tk.Tk()
    try: root.tk.call('tk', 'scaling', 1.3)
    except: pass

    app = GUI(root)
    root.mainloop()