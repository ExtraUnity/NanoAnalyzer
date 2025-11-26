import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMainWindow  
from threading import Event

from src.gui.ui.TrainModelUI import Ui_TrainModel
from src.shared.ModelConfig import ModelConfig
from src.shared.ModelTrainingStats import ModelTrainingStats
from src.model.PlottingTools import plot_loss, plot_difference
class TrainModelWindow(QMainWindow, Ui_TrainModel):
    train_model_signal = QtCore.pyqtSignal(ModelConfig, str, Event)
    training_finished_signal = QtCore.pyqtSignal()

    def __init__(self, update_data_signal, show_testing_difference_signal, mode: str = "train"):
        super().__init__()
        self.setupUi(self)
        self.mode = mode

        self.training_images_directory = None
        self.training_labels_directory = None
        self.test_images_directory = None
        self.test_labels_directory = None
        
        self.training_loss_values = []
        self.validation_loss_values = []
        
        update_data_signal.connect(self.update_loss_values)
        show_testing_difference_signal.connect(self.show_testing_difference)
        self.training_finished_signal.connect(self.on_training_finished)
        self._configure_mode_labels()
        
        self.training_images_button.clicked.connect(self.select_training_images_clicked)
        self.training_labels_button.clicked.connect(self.select_training_labels_clicked)
        self.test_images_button.clicked.connect(self.select_test_images_clicked)
        self.test_labels_button.clicked.connect(self.select_test_labels_clicked)
        self.train_model_button.clicked.connect(self.train_model_clicked)
        self.stop_training_button.clicked.connect(self.stop_training_clicked)

        self.auto_test_set_checkbox.stateChanged.connect(self.auto_test_set_checkbox_clicked)
        self.stop_training_event = Event()
        

    def _configure_mode_labels(self):
        if self.mode == "fine_tune":
            self.title.setText("Fine tune model")
            self.train_model_button.setText("Fine tune")
            self.setWindowTitle("Fine tune model")
        else:
            self.title.setText("Train new model")
            self.train_model_button.setText("Train model")
            self.setWindowTitle("Train new model")


    def open_directory(self, window_text):
        default_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))

        folder_path = QFileDialog.getExistingDirectory(None, window_text, default_path)
        if folder_path:
            return folder_path
        return
    
    def select_training_images_clicked(self):
        folder_path = self.open_directory("Select training images folder")
        if folder_path:
            self.training_images_directory = folder_path
    
    def select_training_labels_clicked(self):
        folder_path = self.open_directory("Select training labels folder")
        if folder_path:
            self.training_labels_directory = folder_path

    def select_test_images_clicked(self):
        folder_path = self.open_directory("Select test images folder")
        if folder_path:
            self.test_images_directory = folder_path

    def select_test_labels_clicked(self):
        folder_path = self.open_directory("Select test labels folder")
        if folder_path:
            self.test_labels_directory = folder_path
    
    def toggle_training_buttons(self):
        self.train_model_button.setEnabled(not self.train_model_button.isEnabled())
        self.stop_training_button.setEnabled(not self.stop_training_button.isEnabled())

    def auto_test_set_checkbox_clicked(self):
        if self.auto_test_set_checkbox.isChecked():
            self.test_images_button.setEnabled(False)
            self.test_labels_button.setEnabled(False)
        else:
            self.test_images_button.setEnabled(True)
            self.test_labels_button.setEnabled(True)

    def train_model_clicked(self):
        # Prevent starting if already training
        if not self.train_model_button.isEnabled():
            return
            
        self.stop_training_button.setEnabled(True)
        self.train_model_button.setEnabled(False)
        model_config = ModelConfig(images_path=self.training_images_directory,
                    masks_path=self.training_labels_directory,
                    epochs=int(self.epochs_input.text()),
                    learning_rate=float(self.learning_rate_input.text()),
                    with_early_stopping=self.early_stopping_checkbox.isChecked(),
                    with_data_augmentation=self.data_augment_checkbox.isChecked(),
                    test_images_path=self.test_images_directory,
                    test_masks_path=self.test_labels_directory)
        import datetime
        log_dir = "data/logs/training_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logging_label.setText(f"Logging to {log_dir}")
        self.stop_training_event.clear()
        self.train_model_signal.emit(model_config, log_dir, self.stop_training_event)
    
    def stop_training_clicked(self):
        self.logging_label.setText("Stopping training...")
        self.stop_training_event.set()
        # Keep buttons disabled until training actually stops
        self.stop_training_button.setEnabled(False)

    def on_training_finished(self):
        """Called when training completes (either finished or stopped)"""
        self.logging_label.setText("")
        self.train_model_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)

    
    def update_loss_values(self, stats: ModelTrainingStats):
        self.training_loss_data_label.setText(str(stats.training_loss))
        self.val_loss_data_label.setText(str(stats.validation_loss))
        self.best_val_loss_data_label.setText(str(stats.best_loss))
        self.current_epoch_data_label.setText(str(stats.epoch))
        self.best_epoch_data_label.setText(str(stats.best_epoch))
        self.training_loss_values.append(stats.training_loss)
        self.validation_loss_values.append(stats.validation_loss)
        plot_loss(self.training_loss_values, self.validation_loss_values)

    def show_testing_difference(self, input, prediction, label, iou, dice_score):
        plot_difference(input, prediction, label, iou, dice_score)
