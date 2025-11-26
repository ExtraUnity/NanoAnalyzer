import sys
import time
from PyQt5.QtWidgets import QApplication

import threading
from src.shared.torch_coordinator import set_preload_complete

def preload_torch():
    try:
        import torch
        from ncempy.io import dm
        import numpy as np
        _ = torch.Tensor([0])  # Force lazy CUDA init
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
    main()
    #from src.model.CrossValidation import cv_kfold
    #cv_kfold("data/medres_images", "data/medres_masks")