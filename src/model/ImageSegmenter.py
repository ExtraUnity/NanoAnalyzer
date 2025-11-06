"""
ImageSegmenter class handles the image segmentation pipeline.
"""

import torch
from src.model.DataTools import ImagePreprocessor
from src.model.SegmentationAnalyzer import SegmentationAnalyzer


class ImageSegmenter:
    """
    Handles the segmentation of images using a trained U-Net model.
    """
    
    def __init__(self, unet_model):
        """
        Initialize the ImageSegmenter.
        
        Args:
            unet_model: The U-Net model to use for segmentation
        """
        self.unet = unet_model
        self.preprocessor = ImagePreprocessor(self.unet.preferred_input_size)
    
    def segment_image(self, image):
        """
        Process an image through the segmentation pipeline.
        
        Args:
            image: The input image to segment (ParticleImage)

        Returns:
            - Segmented image (2d numpy array)
        """
        # 1. Prepare image patches
        tensor, tensor_mirror_filled, patches, stride_length = self.preprocessor.prepare_image_patches(
            image.pil_image, 
            self.unet.device
        )
        
        # 2. Run model inference
        segmentations = self._run_inference(patches, tensor)
        
        # 3. Post-process segmentation output
        segmented_image_2d = self.preprocessor.post_process_segmentation(
            segmentations,
            tensor_mirror_filled,
            tensor,
            stride_length
        )
        
        return segmented_image_2d
    
    def _run_inference(self, patches, tensor):
        """
        Run model inference on image patches.
        
        Args:
            patches: Array of image patches
            tensor: Original tensor for dtype/device info
            
        Returns:
            numpy array: Model predictions
        """
        from src.shared.torch_coordinator import ensure_torch_ready
        ensure_torch_ready()
        
        # Convert patches to tensor
        patches_tensor = torch.tensor(patches, dtype=tensor.dtype, device=tensor.device)
        
        # Optimize model for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet.to(device, memory_format=torch.channels_last).eval()
                
        # Prepare input tensor
        patches_tensor = patches_tensor.to(device, memory_format=torch.channels_last)
        
        # Run inference with appropriate precision
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda"):
                    output = self.unet(patches_tensor)
            else:
                output = self.unet(patches_tensor)
        
        # Convert to numpy for post-processing
        return output.cpu().detach().numpy()
