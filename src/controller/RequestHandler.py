import threading
import os
from src.model.SegmentationAnalyzer import SegmentationAnalyzer
from src.model.PlottingTools import *

class RequestHandler:
    def __init__(self, pre_loaded_model_name=None):
        self.unet = None
        self.segmenter = None
        self.segmentation_analyzer = SegmentationAnalyzer()
        self.model_ready_event = threading.Event()
        self.load_model_async(pre_loaded_model_name)

    def load_model_async(self, model_name):
        def load():            
            from src.model.UNet import UNet
            from src.model.ImageSegmenter import ImageSegmenter
            self.unet = UNet(pre_loaded_model_path=f"src/data/model/{model_name}")
            self.segmenter = ImageSegmenter(self.unet)
            self.model_ready_event.set()
            print("Model ready")

        threading.Thread(target=load, daemon=True).start()
        
    def process_request_train(self, model_config, log_dir, stop_training_event = None, loss_callback = None, test_callback = None):  
        from src.shared.torch_coordinator import ensure_torch_ready
        ensure_torch_ready()
        
        from src.model.CrossValidation import cv_holdout
        from src.model.UNet import UNet

        self.model_ready_event.wait()
        self.unet = UNet()
        self.segmenter.unet = self.unet
        
        evaluation_result = cv_holdout(self.unet, model_config, stop_training_event, loss_callback, test_callback, log_dir)
        return evaluation_result

    def process_request_segment(self, image, output_folder, return_stats=False):
        """
        Process an image through the segmentation pipeline.
        
        Args:
            image: The input image to segment
            output_folder: Folder to save the statistics
            return_stats: Whether to include summary statistics in the return value
            
        Returns:
            Tuple containing:
            - Segmented image (PIL Image)
            - Annotated image (PIL Image) 
            - Table data
            - Histogram figure
            - Stats (optional, only if return_stats is True)
        """
        # Wait for model to be ready
        self.model_ready_event.wait()
        
        # Step 1. Segment image
        segmented_image_2d = self.segmenter.segment_image(image)
        
        # Step 2. Analyze results and generate statistics
        results = self.segmentation_analyzer.analyze_segmentation(segmented_image_2d, image.file_info, output_folder)
        
        if return_stats:
            return results
        else:
            return results[:-1]  # Return without stats

    def process_request_load_model(self, model_path):
        self.model_ready_event.wait()
        self.unet.load_model(model_path)
        self.segmenter.unet = self.unet
        return None
    
    def process_request_test_model(self, test_data_image_dir, test_data_mask_dir, testing_callback = None, log_file_path = None):
        from src.shared.torch_coordinator import ensure_torch_ready
        ensure_torch_ready()
        
        from src.model.SegmentationDataset import SegmentationDataset
        from torch.utils.data import DataLoader


        dataset = SegmentationDataset(test_data_image_dir, test_data_mask_dir)
        test_dataloader = DataLoader(dataset, batch_size=1)
        from src.model.ModelEvaluator import ModelEvaluator

        self.model_ready_event.wait()
        
        # Generate default log file path if not provided
        if log_file_path is None:
            import os
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.path.dirname(test_data_image_dir), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_file_path = os.path.join(logs_dir, f"evaluation_results_{timestamp}.txt")
        
        evaluation_result = ModelEvaluator.evaluate_model(self.unet, test_dataloader, testing_callback, log_file_path)
        return evaluation_result
        
    def process_request_segment_folder(self, input_folder, output_parent_folder):
        """
        Process all images in a folder through the segmentation pipeline.
        
        Args:
            input_folder: Path to folder containing images to process
            output_parent_folder: Path to folder where results will be saved
        """
        from src.model.BatchProcessor import BatchProcessor
        
        self.model_ready_event.wait()
        
        batch_processor = BatchProcessor()
        batch_processor.process_folder(
            input_folder,
            output_parent_folder,
            self.process_request_segment
        )
        
    def process_request_load_image(self, image_path):
        """
        Load and preprocess an image for segmentation.
        
        Args:
            image_path: Path to the image file to load
            
        Returns:
            ParticleImage: The loaded and preprocessed image
        """
        from src.shared.ParticleImage import ParticleImage
        return ParticleImage.load_and_preprocess(image_path)