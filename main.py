import os
import sys
import time
import threading
import multiprocessing
from PyQt5.QtWidgets import QApplication

from src.shared.torch_coordinator import set_preload_complete

def preload_torch():
    try:
        import torch
        from ncempy.io import dm
        import numpy as np
        _ = torch.Tensor([0])  # Force lazy CUDA init
        torch.set_num_threads(max(1, os.cpu_count()//4)*3)      
        torch.set_num_interop_threads(1)

    except Exception as e:
        print(f"Error during torch preloading: {e}")
    finally:
        set_preload_complete()  # Signal that preloading is done

def main():
    # Start preloading in background
    threading.Thread(target=preload_torch, daemon=True).start()
    
    # Don't wait - immediately start GUI
    from src.gui.windows.MainWindow import MainWindow
    
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.MainWindow.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    # Required when frozen with PyInstaller to prevent child worker processes
    # from re-launching the GUI when torch DataLoader spawns workers.
    multiprocessing.freeze_support()
    main()
    #from src.model.CrossValidation import cv_kfold
    #cv_kfold("data/medres_images", "data/medres_masks")
