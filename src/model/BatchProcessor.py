import os
from src.model.StatsWriter import StatsWriter
from src.shared.ParticleImage import ParticleImage

class BatchProcessor:
    """Handles batch processing of multiple images for segmentation."""

    def process_folder(self, input_folder, output_parent_folder, process_single_image_func):
        """
        Process all images in a folder through the segmentation pipeline.
        
        Args:
            input_folder: Path to folder containing images to process
            output_parent_folder: Path to folder where results will be saved
            process_single_image_func: Function that processes a single image and returns results
            
        Returns:
            None
            
        Raises:
            ValueError: If any image in the folder has 'pixel' as its unit
        """
        all_stats = []
        all_file_info = []
        images = []
        file_paths = []
        # First validate all images have proper units
        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)
            image = ParticleImage.load_and_preprocess(file_path)
            if image.file_info.unit.lower() == "pixel":
                raise ValueError(f"Image '{filename}' has no readable physical unit (unit is 'pixel'). Cannot process folder when physical units are missing.")
            images.append(image)
            file_paths.append(file_path)

        # Process each image in the input folder
        for image, file_path in zip(images, file_paths):            
            # Create output folder for this image
            output_folder = os.path.join(output_parent_folder, image.file_info.file_name)
            os.makedirs(output_folder, exist_ok=True)
            
            # Process the image
            segmented_image_pil, annotated_image_pil, _, _, stats = process_single_image_func(image, output_folder, True)
            
            # Save results
            self._save_output_images(
                segmented_image_pil, 
                annotated_image_pil, 
                output_folder, 
                image.file_info.file_name
            )
            
            all_stats.append(stats)
            all_file_info.append(image.file_info)

        # Write combined statistics
        self._write_combined_stats(all_stats, all_file_info, output_parent_folder)
    
    def _save_output_images(self, segmented_image, annotated_image, output_folder, file_name):
        """Save segmented and annotated images to disk."""
        segmented_image.save(os.path.join(output_folder, f"{file_name}_segmented.tif"))
        annotated_image.save(os.path.join(output_folder, f"{file_name}_annotated.tif"))
    
    def _write_combined_stats(self, all_stats, all_file_info, output_folder):
        """Write combined statistics for all processed images."""
        stats_writer = StatsWriter()
        stats_writer.write_all_stats_to_txt(all_stats, all_file_info, output_folder)