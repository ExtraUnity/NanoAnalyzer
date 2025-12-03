import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from src.model.StatsWriter import StatsWriter
from src.shared.FileInfo import FileInfo
class SegmentationAnalyzer():

    def get_connected_components(self, image):
        num_labels, labels, area_stats, centroids= cv2.connectedComponentsWithStats(image)
        return num_labels, labels, area_stats, centroids
    
    def analyze_segmentation(self, segmented_image_2d, file_info, output_folder):
        """
        Analyze the segmentation and generate output images and statistics.
        
        Args:
            segmented_image_2d: The segmented image as a 2D array
            file_info: FileInfo object containing image metadata
            output_folder: Folder to save the statistics
            
        Returns:
            Tuple containing:
            - Segmented image (PIL Image)
            - Annotated image (PIL Image)
            - Table data
            - Histogram figure
            - Stats
        """
        # Get components
        num_labels, _, stats, centroids = self.get_connected_components(segmented_image_2d)
        particle_count = num_labels - 1
        
        # Create output images
        annotated_image = self.add_annotations(segmented_image_2d, centroids)
        annotated_image_pil = Image.fromarray(annotated_image)
        segmented_image_pil = Image.fromarray(segmented_image_2d)
        
        # Generate analysis outputs
        table_data = self.format_table_data(stats, file_info, particle_count)
        histogram_fig = self.create_histogram(stats, file_info)
        
        # Save statistics
        from src.model.StatsWriter import StatsWriter
        stats_writer = StatsWriter()
        stats_writer.write_stats_to_txt(stats, file_info, particle_count, output_folder)
        
        return segmented_image_pil, annotated_image_pil, table_data, histogram_fig, stats

    def save_histogram_as_image(self, fig):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        hist_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "histogram", "diameter_histogram.png")
        os.makedirs(os.path.dirname(hist_image_path), exist_ok=True)
        fig.savefig(hist_image_path)
        plt.close()
        histogram_image = Image.open(hist_image_path)
        return histogram_image
    
    def create_histogram(self, stats, file_info: FileInfo):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        stats_writer = StatsWriter()
        try:
            scaled_areas, scaled_diameters = stats_writer.get_scaled_meassurements(stats, file_info)
            histogram_data = {
                "Area": scaled_areas,
                "Diameter": scaled_diameters
            }

            rice_rule_steps = int(np.ceil(2 * len(histogram_data["Diameter"]) ** (1 / 3))) 

            fig, ax = plt.subplots()            
            ax.hist(
                histogram_data["Diameter"], 
                bins=rice_rule_steps, 
                label="Diameter", 
                edgecolor='black'
                )
            ax.set_title("Particle Diameter Histogram")
            ax.set_xlabel("Diameter"+" ["+file_info.unit+"]")
            ax.set_ylabel("Frequency")
            ax.legend(title=f"Steps: {rice_rule_steps} (Rice-rule)")
            
            self.save_histogram_as_image(fig)
            return fig           
        except Exception as e:
            print("Error in creating histogram: ", e)
            return None
        
    def add_annotations(self, image, centroids, min_distance=10, max_offset_attempts=5):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            font_scale = 0.3
            thickness = 1
            used_positions = []

            for i in range(1, len(centroids)):
                cX, cY = int(centroids[i][0]), int(centroids[i][1])
                label = str(i)
                final_x, final_y = self.check_particle_distance(cX, cY, used_positions, min_distance, max_offset_attempts)
                used_positions.append((final_x, final_y))
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = final_x - text_width // 2
                text_y = final_y + text_height // 2
                cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

            return image_rgb

        except Exception as e:
            print("Error in add_annotations: ", e)
            return image
    
    def check_particle_distance(self, cX, cY, used_positions, min_distance=10, max_offset_attempts=5):
        offset_attempt = 0
        final_x, final_y = cX, cY

        while offset_attempt < max_offset_attempts:
            too_close = False
            for prev_x, prev_y in used_positions:
                distance = ((final_x - prev_x) ** 2 + (final_y - prev_y) ** 2) ** 0.5
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                return final_x, final_y

            offset_attempt += 1
            offset_amount = offset_attempt * 5 
            final_x = cX + offset_amount
            final_y = cY - offset_amount
        return final_x, final_y

    
    

    def format_table_data(self, stats: np.ndarray, file_info: FileInfo, particle_count: int):
        if particle_count == 0:
            return {
                "Count":    [0, 0, 0, 0],  
                "Area":     [0, 0, 0, 0],  
                "Diameter": [0, 0, 0, 0]
            }
            
        stats_writer = StatsWriter()
        scaled_areas, scaled_diameters = stats_writer.get_scaled_meassurements(stats, file_info)
        
        area_mean = np.mean(scaled_areas).round(2)
        area_max = np.max(scaled_areas).round(2)
        area_min = np.min(scaled_areas).round(2)
        area_std = np.std(scaled_areas).round(2)
        
        diameter_mean = np.mean(scaled_diameters).round(2)
        diameter_max = np.max(scaled_diameters).round(2)
        diameter_min = np.min(scaled_diameters).round(2)
        diameter_std = np.std(scaled_diameters).round(2)


        unit = " "+file_info.unit
        table_data = {
        "Count":    [particle_count, particle_count, particle_count, particle_count],  
        "Area":    [str(area_mean)+unit+"²", str(area_min)+unit+"²", str(area_max)+unit+"²", str(area_std)+unit+"²"],
        "Diameter":    [str(diameter_mean)+unit, str(diameter_min)+unit, str(diameter_max)+unit, str(diameter_std)+unit]
        }

        return table_data
