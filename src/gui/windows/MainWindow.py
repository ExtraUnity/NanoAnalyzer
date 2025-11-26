from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import (
    QFileDialog, QMainWindow, QMessageBox, QApplication, 
    QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout,
    QDialog, QLabel, QAction, QToolBar
)
#from PIL import ImageQt
from PyQt5.QtCore import QDir
import os
import csv
import threading
from threading import Event
from functools import partial 

from src.gui.ui.MainUI import Ui_MainWindow
from src.controller.Controller import Controller
from src.shared.Commands import Command
from src.gui.windows.TrainModelWindow import TrainModelWindow
from src.gui.windows.SetScaleWindow import SetScaleWindow
from src.shared.ModelConfig import ModelConfig
from src.gui.TableData import TableData
from src.shared.ModelTrainingStats import ModelTrainingStats
from PIL import Image
from PIL.ImageQt import ImageQt
from src.gui.windows.MessageBoxes import *
from src.model.PlottingTools import plot_loss, plot_difference
from src.shared.Formatters import _truncate
from src.shared.ParticleImage import ParticleImage
from src.shared.IOFunctions import validate_file_extension
from src.shared.EvaluationResult import EvaluationResult
from PyQt5.QtCore import QObject, pyqtSignal, QThread

class SegmentationWorker(QObject):
    """Worker that runs segmentation via controller.safe_request in a separate thread."""
    finished = pyqtSignal(object)  # emits result or None on error
    error = pyqtSignal(str)

    def __init__(self, controller_owner, command, *args, **kwargs):
        super().__init__() 
        self.controller_owner = controller_owner
        self.command = command
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            res = self.controller_owner.safe_request(self.command, *self.args, **self.kwargs)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow, Ui_MainWindow):
    update_train_model_values_signal = QtCore.pyqtSignal(ModelTrainingStats)
    show_testing_difference_signal = QtCore.pyqtSignal(object, object, object, object, object)
    errorOccurred = QtCore.pyqtSignal(str, object)
    segmentation_finished = pyqtSignal(object)
    segmentation_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.MainWindow = QMainWindow()
        self.setupUi(self.MainWindow)
        preloaded_model_name = "UNet_best_16-10.pt"
        self.current_model_label.setText(f"{preloaded_model_name}")
        self.controller = Controller(preloaded_model_name)
        self.image_path = None
        self.image: ParticleImage = None
        self.segmented_image = None
        self.annotated_image = None
        self.file_path_image = None
        self.scale_is_selected = False
        self.table_data_set = False
        self.scale_input_set = False
        self.show_annotated_image = True  
        self.graphicsView_scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.graphicsView_scene)
        self.training_state = "not done"
        self.standard_model_config = ModelConfig(images_path="data/images",
                                                 masks_path="data/masks",
                                                 epochs=5,
                                                 learning_rate=0.0005,
                                                 with_early_stopping=True,
                                                 with_data_augmentation=True)
        
        self.train_model_window = None
        self.fine_tune_window = None
        self.train_thread = None
        self.training_loss_values = []
        self.validation_loss_values = []
        self._setup_model_toolbar()

        self.plot_segmentation_scene = QGraphicsScene(self)
        self.plot_segmentation.setScene(self.plot_segmentation_scene)
        self.plot_graph_layout = QVBoxLayout(self.plot_histogram)
        self.plot_histogram.setLayout(self.plot_graph_layout)


        # Other connections
        self.action_test_model.triggered.connect(self.on_test_model_clicked)
        self.action_open_image.triggered.connect(self.on_open_image_clicked)
        self.actionRun_Segmentation_on_Current_Image.triggered.connect(self.on_segment_image_clicked)
        self.actionRun_Segmentation_on_folder.triggered.connect(self.on_segment_folder_clicked)
        self.action_load_model.triggered.connect(self.on_load_model_clicked)
        self.actionExport_Segmentation_2.triggered.connect(self.on_export_segmented_clicked)
        self.actionExport_Data_as_csv.triggered.connect(self.on_export_table_data_csv_clicked)
        self.actionExport_Statistics_as_CSV.triggered.connect(self.on_export_statistics_clicked)
        self.action_new_data_train_model.triggered.connect(self.on_train_model_custom_data_clicked)
        self.fullscreen_image_button.clicked.connect(self.on_fullscreen_image_clicked)
        self.radioButton.toggled.connect(self.on_toggle_segmented_image_clicked)
        self.setScaleButton.clicked.connect(self.open_set_scale_window) 
        self.runSegmentationBtn.clicked.connect(self.on_segment_image_clicked)

        self.errorOccurred.connect(self.handle_error)
        # Segmentation thread / worker references
        self._seg_thread = None
        self._seg_worker = None
        # Connect signals to handlers
        self.segmentation_finished.connect(self._on_segmentation_finished)
        self.segmentation_failed.connect(self._on_segmentation_failed)

    def _setup_model_toolbar(self):
        self.model_toolbar = QToolBar("Model", self.MainWindow)
        self.model_toolbar.setObjectName("model_toolbar")
        self.MainWindow.addToolBar(self.model_toolbar)
        self.model_toolbar.addAction(self.action_new_data_train_model)

        self.action_fine_tune_model = QAction("Fine tune model", self.MainWindow)
        self.action_fine_tune_model.setObjectName("action_fine_tune_model")
        self.model_toolbar.addAction(self.action_fine_tune_model)
        self.menu_model.addAction(self.action_fine_tune_model)
        self.action_fine_tune_model.triggered.connect(self.on_fine_tune_model_clicked)


    def handle_error(self, message, handler=None):
        messageBox(self, message)
        if handler:
            handler()

    def safe_request(self, command, *args, on_error=None, **kwargs):
        from src.shared.RequestError import RequestError
        res = self.controller.process_command(command, *args, **kwargs)
        if isinstance(res, RequestError):
            print(f"Error: {res.message}")
            self.errorOccurred.emit(res.message, on_error)
            return None
        return res

    def open_set_scale_window(self):
        if (self.image_path == None or self.image == None):
            messageBox(self, "No image found")
            return
        self.set_scale_window = SetScaleWindow(self.image, self.file_path_image, overlay_updater=self)
        self.set_scale_window.show()
            
    def display_image_metadata_overlay(self, file_path: str):
        file_name = os.path.basename(file_path)

        image_width = str(self.image.file_info.width)
        image_height = str(self.image.file_info.height)
        image_real_width = str(_truncate(self.image.file_info.real_width, 2))
        image_real_height = str(_truncate(self.image.file_info.real_height, 2))

        metadata = (
            f"{image_real_width}x{image_real_height} {self.image.file_info.unit} "
            f"({image_width}x{image_height})"
        )

        html_text = f"""
        <table width="100%" cellspacing="0" cellpadding="0">
            <tr>
                <td align="left">{file_name}</td>
                <td align="right">{metadata}</td>
            </tr>
        </table>
        """

        self.input_image_metadata.setText(html_text)

    
    
    def toggle_count_overlay(self):
            if self.segmented_image is None or self.annotated_image is None:
                messageBox(self, "No segmented image available to toggle.")
                return

            self.show_annotated_image = not self.show_annotated_image
            self.update_segmented_image_view()
        
    def on_fullscreen_image_clicked(self):
        if (self.image_path == None):
            messageBox(self, "Fullscreen failed: No image found")
            return
        import matplotlib
        matplotlib.use('Qt5Agg') 
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)
        
        manager = plt.get_current_fig_manager()

        axes[0].imshow(self.image.pil_image, cmap='gray')
        axes[0].set_title("Image")


        if self.show_annotated_image and self.annotated_image is not None:
            axes[1].imshow(self.annotated_image, cmap='gray')
            axes[1].set_title("Annotated Segmentation")  
        elif self.segmented_image is not None:
            axes[1].imshow(self.segmented_image, cmap='gray')
            axes[1].set_title("Segmentation")

        manager.window.showMaximized()
        plt.tight_layout()
        plt.pause(0.1)


    def set_table_data(self, table_data):
        data = TableData(table_data)
        data.insertIn(self.table_widget)
        self.table_data_set = True

    def on_segment_folder_clicked(self):
        input_folder_path = QFileDialog.getExistingDirectory(None, "Select an input folder", "")
        output_folder_path = QFileDialog.getExistingDirectory(None, "Select a folder for the output", "")
        if input_folder_path and output_folder_path:
            # Run segmentation in background thread; helper handles wiring/cleanup
            self.set_ui_busy(True)
            self._start_segmentation_thread(Command.SEGMENT_FOLDER, self._on_segmentation_finished, input_folder_path, output_folder_path)
            self._seg_thread.start()
        else:
            messageBox(self, "Error in uploading directory")
            return

    def on_train_model_custom_data_clicked(self):
        self.train_model_window = TrainModelWindow(self.update_train_model_values_signal, self.show_testing_difference_signal)
        self.train_model_window.train_model_signal.connect(self.train_model_custom_data)
        self.train_model_window.show()

    def on_fine_tune_model_clicked(self):
        self.fine_tune_window = TrainModelWindow(self.update_train_model_values_signal, self.show_testing_difference_signal, mode="fine_tune")
        self.fine_tune_window.train_model_signal.connect(self.fine_tune_model_custom_data)
        self.fine_tune_window.show()


    def train_model_custom_data(self, model_config: ModelConfig, log_dir: str, stop_training_event: Event):
        result = confirmTrainingMessageBox(self, "Training a new model may take a while, do you want to continue?")
        if result == QMessageBox.No:
            return

        self._start_segmentation_thread(Command.RETRAIN, self.show_metrics_popup, 
                model_config,
                log_dir,
                stop_training_event,
                self.update_training_model_stats,
                self.show_testing_difference,
                on_error=lambda: self.train_model_window.training_finished_signal.emit())
        self._seg_thread.start()  

    def fine_tune_model_custom_data(self, model_config: ModelConfig, log_dir: str, stop_training_event: Event):
        result = confirmTrainingMessageBox(self, "Fine tuning will update the currently loaded model. Continue?")
        if result == QMessageBox.No:
            return

        self._start_segmentation_thread(Command.FINE_TUNE, self.show_metrics_popup, 
                model_config,
                log_dir,
                stop_training_event,
                self.update_training_model_stats,
                self.show_testing_difference,
                on_error=lambda: self.fine_tune_window.training_finished_signal.emit())
        self._seg_thread.start()  


    def update_training_model_stats(self, stats: ModelTrainingStats):
        self.update_train_model_values_signal.emit(stats)

    def show_testing_difference(self, input, prediction, label, iou, dice):
        self.show_testing_difference_signal.emit(input, prediction, label, iou, dice)

    def update_loss_values(self, stats: ModelTrainingStats):
        # Update the GUI with the training stats
        self.training_loss_values.append(stats.training_loss)
        self.validation_loss_values.append(stats.validation_loss)
        plot_loss(self.training_loss_values, self.validation_loss_values)
    
    def clear_image(self):
        self.graphicsView_scene.clear()
        self.plot_segmentation_scene.clear()
        self.image_path = None
        self.scale_is_selected = False
        self.scale_input_set = False
        self.segmented_image = None

    def on_open_image_clicked(self):
        default_image_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'images'))
        
        self.scale_is_selected = False
        self.scale_input_set = False
        
        self.file_path_image, _ = QFileDialog.getOpenFileName(
            self, 
            "Select an image file", 
            default_image_path, 
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.dm3 *.dm4);;All Files (*)")
        

        if self.file_path_image: 
            if not validate_file_extension(self.file_path_image, [".png", ".jpg", "jpeg", ".tif", ".tiff", ".dm3", ".dm4"]):
                messageBox(self, "Error in uploading file: Unsupported file format")
                return
            self.clear_image()
            self.image_path = self.file_path_image
            
            res = self.safe_request(Command.LOAD_IMAGE, self.file_path_image) 
            if res is None:
                return
            self.image = res
            pixmap = self.load_pixmap(self.image.pil_image)
            pixmap_item = QGraphicsPixmapItem(pixmap.scaled(500, 500, aspectRatioMode=1))
            self.graphicsView_scene.clear()
            self.graphicsView_scene.addItem(pixmap_item)
            self.display_image_metadata_overlay(self.file_path_image)
        else:
            messageBox(self, "Error in uploading file: No file selected")

    def load_pixmap(self, image: Image) -> QPixmap:
        qimage = ImageQt(image)
        return QPixmap.fromImage(qimage) 


    def show_testing_difference_mainwindow(self, prediction, label, iou, dice_score):
        from src.model.PlottingTools import plot_difference
        plot_difference(prediction, label, iou, dice_score)

    def show_metrics_popup(self, evaluation_result: EvaluationResult):
        self.set_ui_busy(False)
        if self.train_model_window is not None:
            self.train_model_window.training_finished_signal.emit()
        if self.fine_tune_window is not None:
            self.fine_tune_window.training_finished_signal.emit()
            
        if evaluation_result is None:
            return
        mean_iou = evaluation_result.mean_iou
        mean_dice = evaluation_result.mean_dice
        dialog = QDialog()
        dialog.setWindowTitle("Model Evaluation Metrics")
        dialog.resize(400, 400)  # width, height

        layout = QVBoxLayout()
        label = QLabel(f"<h3>Mean IOU:</h3> {mean_iou:.4f}<br>"
                       f"<h3>Mean Dice Score:</h3> {mean_dice:.4f}<hr>"
                       f"<h3>Min IOU:</h3> {evaluation_result.min_iou:.4f}<br>"
                       f"<h3>Min Dice Score:</h3> {evaluation_result.min_dice:.4f}<hr>"
                       f"<h3>Max IOU:</h3> {evaluation_result.max_iou:.4f}<br>"
                       f"<h3>Max Dice Score:</h3> {evaluation_result.max_dice:.4f}<hr>"
                       f"<h3>Tested on:</h3> {len(evaluation_result)} samples"
                       )
        label.setWordWrap(True)

        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.exec_()


    def on_test_model_clicked(self):
        image_folder_path = QFileDialog.getExistingDirectory(None, "Select test images folder", "")
        mask_folder_path = QFileDialog.getExistingDirectory(None, "Select test masks folder", "")

        if image_folder_path and mask_folder_path:
            try:
                self.show_testing_difference_signal.disconnect()
            except TypeError:
                pass
            self.show_testing_difference_signal.connect(plot_difference)
            self.set_ui_busy(True)
            self._start_segmentation_thread(Command.TEST_MODEL, self.show_metrics_popup, image_folder_path, mask_folder_path, self.show_testing_difference)
            self._seg_thread.start()            
        else:
            messageBox(self, "Error in uploading directories")
            return

    def on_segment_image_clicked(self):
        if (self.image_path == None):
            messageBox(self, "Segmentation failed: No image found")
            return
        # Run segmentation in background thread; helper handles wiring/cleanup
        self.set_ui_busy(True)
        self._start_segmentation_thread(Command.SEGMENT, self._on_segmentation_finished,self.image, "data/statistics")
        self._seg_thread.start()

    def _start_segmentation_thread(self, command=Command.SEGMENT, on_finished_function=None, *args, **kwargs):
        """Prepare a QThread and SegmentationWorker for running segmentation."""
        # Ensure previous thread is cleaned up
        self._cleanup_seg_thread()

        self._seg_thread = QThread()
        self._seg_worker = SegmentationWorker(self, command, *args, **kwargs)
        self._seg_worker.moveToThread(self._seg_thread)

        # Wire up signals
        self._seg_thread.started.connect(self._seg_worker.run)
        
        try:
            self.segmentation_finished.disconnect()
        except TypeError:
            pass

        self.segmentation_finished.connect(on_finished_function)
        self.segmentation_failed.connect(self._on_segmentation_failed)

        self._seg_worker.finished.connect(lambda res: self.segmentation_finished.emit(res))
        self._seg_worker.error.connect(lambda msg: self.segmentation_failed.emit(msg))

        # Ensure thread is cleaned after finish/fail
        self._seg_worker.finished.connect(self._seg_thread.quit)
        self._seg_worker.finished.connect(self._seg_worker.deleteLater)
        self._seg_worker.error.connect(self._seg_thread.quit)
        self._seg_worker.error.connect(self._seg_worker.deleteLater)
        self._seg_thread.finished.connect(self._seg_thread.deleteLater)

    def _cleanup_seg_thread(self):
        """Try to clean up any existing segmentation QThread."""
        if getattr(self, '_seg_thread', None) is not None:
            try:
                # Ask the existing thread to quit
                self._seg_thread.quit()
                
                # For training threads, we need to wait longer
                # Check if thread is still running and wait appropriately
                if self._seg_thread.isRunning():
                    # Wait up to 5 seconds for thread to stop gracefully
                    if not self._seg_thread.wait(timeout=5000):
                        # If thread doesn't stop gracefully, terminate it
                        print("Warning: Training thread did not stop gracefully, terminating...")
                        self._seg_thread.terminate()
                        # Give it a moment to terminate
                        self._seg_thread.wait(timeout=1000)
                        
            except Exception as e:
                print(f"Error during thread cleanup: {e}")
                
        self._seg_thread = None
        self._seg_worker = None
        
    def display_histogram(self, histogram_fig):
        if histogram_fig is not None:
            if hasattr(self, 'histogram_canvas') and self.histogram_canvas:
                self.histogram_canvas.setParent(None)
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            self.histogram_canvas = FigureCanvas(histogram_fig)
            self.plot_graph_layout.addWidget(self.histogram_canvas)
            self.histogram_canvas.draw()
            
        
    def on_toggle_segmented_image_clicked(self):
        if self.segmented_image is None or self.annotated_image is None:
            messageBox(self, "No segmented image available to toggle.")
            return

        self.show_annotated_image = not self.show_annotated_image
        self.update_segmented_image_view()

    def update_segmented_image_view(self):
        if self.show_annotated_image:
            image_to_display = self.annotated_image
        else:
            image_to_display = self.segmented_image
            
        image_temp = ImageQt(image_to_display)
        pixmap = QPixmap.fromImage(image_temp)
        pixmap_item = QGraphicsPixmapItem(pixmap.scaled(500, 500, aspectRatioMode=1))
        self.plot_segmentation_scene.clear()
        self.plot_segmentation_scene.addItem(pixmap_item)

    # UI busy helpers
    def set_ui_busy(self, busy: bool):
        """Disable/enable central interactive widgets while background work is running."""
        widgets = [
            self.runSegmentationBtn,
            self.setScaleButton,
            self.fullscreen_image_button,
            self.actionRun_Segmentation_on_Current_Image,
            self.actionRun_Segmentation_on_folder,
            self.action_open_image,
            self.action_load_model,
            self.action_new_data_train_model,
            self.action_fine_tune_model,
            self.action_test_model,
            self.actionExport_Segmentation_2,
            self.actionExport_Data_as_csv,
            self.actionExport_Statistics_as_CSV,
        ]
        for w in widgets:
            try:
                # QAction uses setEnabled, widgets too
                w.setEnabled(not busy)
            except Exception:
                pass

    def _on_segmentation_finished(self, res):
        # Re-enable UI
        self.set_ui_busy(False)
        if res is None:
            return
        
        try:
            self.segmented_image, self.annotated_image, table_data, histogram_fig = res
            self.set_table_data(table_data)
            self.update_segmented_image_view()
            self.display_histogram(histogram_fig)
        except Exception as e:
            messageBox(self, f"Failed processing segmentation result: {e}")

    def _on_segmentation_failed(self, message):
        self.set_ui_busy(False)
        messageBox(self, f"Segmentation failed: {message}")
            

    def on_load_model_clicked(self):        
        default_models_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'models'))
        file_path, selected_filter = QFileDialog.getOpenFileName(
            None, 
            "Select a model file", 
            default_models_path, 
            "PT Files (*.pt);;All Files (*)"
            )
        
        if file_path: 
            if "PT" in selected_filter:
                if not file_path.endswith(".pt"):  
                    file_path += ".pt"
                self.safe_request(Command.LOAD_MODEL, file_path) 
                self.current_model_label.setText(f"{os.path.basename(file_path)}")
                messageBox(self, "success", "Model loaded successfully")
            else:
                messageBox(self, "Error: The selected file is not a PT file.")


    def on_export_segmented_clicked(self):
        if(self.segmented_image == None):
            messageBox(self, "Export failed: No segmented image was found to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            None, 
            "Save Image", 
            "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")

        if file_path: 
            if not os.path.splitext(file_path)[1]: 
                if "PNG" in selected_filter:
                    file_path += ".png"
                elif "JPEG" in selected_filter:
                    file_path += ".jpg"
                else:
                    file_path += ".png" 
            
            self.segmented_image.save(file_path)            
            messageBox(self, "success", "Segmented image exported successfully")
        else:
            messageBox(self, "Error: File path is not selected.")
               # Check if the table is empty
 
            
        
          
    def on_export_table_data_csv_clicked(self):
        if not self.table_data_set:
            messageBox(self, f"Failed to export data: No table data found")
            return

        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Save CSV")
        file_dialog.setNameFilter("CSV Files (*.csv);;All Files (*)")
        file_path, selected_filter = file_dialog.getSaveFileName(
            options=QFileDialog.Options()
        )

        # Center the dialog (optional, after showing it)
        dialog_geometry = file_dialog.geometry()
        screen_geometry = QApplication.desktop().screenGeometry() 
        screen_center = screen_geometry.center()
        dialog_center = dialog_geometry.center()
        file_dialog.move(screen_center - dialog_center)

        # If user canceled
        if not file_path:
            return

        # Ensure .csv extension
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                headers = [
                    self.table_widget.horizontalHeaderItem(col).text()
                    for col in range(self.table_widget.columnCount())
                ]
                writer.writerow(headers)

                for row in range(self.table_widget.rowCount()):
                    row_data = [
                        self.table_widget.item(row, col).text()
                        if self.table_widget.item(row, col) else ""
                        for col in range(self.table_widget.columnCount())
                    ]
                    writer.writerow(row_data)

            messageBox(self, "success", "Data exported successfully")
        except Exception as error:
            messageBox(self, f"Failed to export data: {str(error)}")


    def parse_fixed_width_line(self, line):
        # Adds units to the header metrics
        return [
            line[0:12].strip(),
            line[12:32].strip(),
            line[32:52].strip()
        ]
        
    def on_export_statistics_clicked(self):
        if self.image is None:
            messageBox(self, "Export failed: No image found")
            return

        # OBS: Uses the txt file generated during segmentation
        txt_path = os.path.abspath(os.path.join("data", "statistics", f"{self.image.file_info.file_name}_statistics.txt"))
        file_path, selected_filter = QFileDialog.getSaveFileName(
        None,
        "Save Statistics as CSV",
        QDir.homePath(), 
        "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            messageBox(self, "Error: No save path selected.")
            return

        if not os.path.splitext(file_path)[1]:
            file_path += ".csv"

        try:
            # Parse particle data from TXT
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                header = None
                data_started = False
                particle_data = []

                for line in lines:
                    if 'Particle No.' in line:
                        header = self.parse_fixed_width_line(line)
                        data_started = True
                        continue
                    if '_______________________________' in line:
                        break
                    if data_started:
                        parts = self.parse_fixed_width_line(line)
                        if len(parts) == 3:
                            particle_data.append(parts)
                        
            # Write to CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)                 
                writer.writerows(particle_data)

            messageBox(self, "success", "Statistics exported successfully")
        except Exception as e:
            messageBox(self, f"Export failed: {str(e)}")
            
