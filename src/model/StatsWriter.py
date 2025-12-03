from src.shared.FileInfo import FileInfo
import os

class StatsWriter:
    def __init__(self):
        # Conversion factors to nanometers (nm)
        self._unit_conversion_factors = {
            "nm": 1,
            "\u00B5m": 1000,  # micrometer
            "mm": 1000000,
            "cm": 10000000,
            "inch": 25400000
        }
        
    def _convert_measurement(self, value, from_unit, to_unit):
        """Convert a measurement from one unit to another.
        
        Args:
            value: The value to convert
            from_unit: The unit to convert from
            to_unit: The unit to convert to
            
        Returns:
            float: The converted value
        """
        if from_unit == to_unit:
            return value
        
        # Convert to nanometers first
        nm_value = value * self._unit_conversion_factors.get(from_unit, 1)
        # Then convert to target unit
        return nm_value / self._unit_conversion_factors.get(to_unit, 1)
    
    def __get_diameters(self, stats):
        """Calculate diameters using Equivalent Circular Diameter (ECD)."""
        import numpy as np
        from cv2 import CC_STAT_AREA
        diameters = np.empty(stats.shape[0]-1)
        for label_idx in range(1, stats.shape[0]):
            area = stats[label_idx, CC_STAT_AREA]
            diameter = 2 * np.sqrt(area / np.pi)
            diameters[label_idx-1] = diameter
        return diameters
    
    def __get_pixel_areas(self, stats):
        from cv2 import CC_STAT_AREA
        return stats[1:, CC_STAT_AREA] 
    
    def get_scaled_meassurements(self, stats, file_info: FileInfo):
        pixel_area = file_info.pixel_width * file_info.pixel_height
        scaled_areas = self.__get_pixel_areas(stats) * file_info.downsize_factor**2 * pixel_area
        scaled_diameters = self.__get_diameters(stats) * file_info.downsize_factor * file_info.pixel_width 
        return scaled_areas, scaled_diameters


    def write_stats_to_txt(self, stats, file_info: FileInfo, particle_count, output_folder):
        """Write statistics for a single image to a separate TXT file."""
        try:
            scaled_areas, scaled_diameters = self.get_scaled_meassurements(stats, file_info)
            txtfile_path = os.path.join(output_folder, f"{file_info.file_name}_statistics.txt")
            os.makedirs(os.path.dirname(txtfile_path), exist_ok=True)

            with open(txtfile_path, "w", encoding="utf-8") as txtfile:
                self._write_header(txtfile, file_info.unit)
                self._write_particle_data(txtfile, scaled_areas, scaled_diameters, particle_count, file_info.unit)
        except Exception as e:
            print("Error in writing statistics to txt file: ", e)

    def _get_smallest_unit(self, units):
        """Find the smallest unit from a list of units based on conversion factors.
        Smaller unit = smaller conversion factor (e.g., nm has smaller factor than μm)"""
        min_factor = float("inf")
        smallest_unit = None
        for unit in units:
            if unit.lower() == "pixel":
                continue
            factor = self._unit_conversion_factors.get(unit, 0)
            if factor < min_factor:
                min_factor = factor
                smallest_unit = unit
        return smallest_unit

    def write_all_stats_to_txt(self, stats_list, file_info_list, output_folder, output_filename="all_statistics.txt"):
        """Write statistics for ALL images combined into one TXT file.
        Converts all measurements to the smallest unit found among the images."""
        import numpy as np
        try:
            all_areas = []
            all_diameters = []
            txtfile_path = os.path.join(output_folder, output_filename)
            os.makedirs(os.path.dirname(txtfile_path), exist_ok=True)
            
            # Find the smallest unit among all images
            units = [info.unit for info in file_info_list]
            target_unit = self._get_smallest_unit(units)
            if not target_unit:
                raise ValueError("No valid physical units found in images")
            
            with open(txtfile_path, "w", encoding="utf-8") as txtfile:
                self._write_header(txtfile, target_unit)

                global_particle_idx = 1
                for stats, file_info in zip(stats_list, file_info_list):
                    # Get measurements in the original unit
                    scaled_areas, scaled_diameters = self.get_scaled_meassurements(stats, file_info)
                    
                    # Convert to target unit if needed
                    if file_info.unit != target_unit:
                        scaled_areas = [self._convert_measurement(area, file_info.unit, target_unit) for area in scaled_areas]
                        scaled_diameters = [self._convert_measurement(diam, file_info.unit, target_unit) for diam in scaled_diameters]

                    for area, diameter in zip(scaled_areas, scaled_diameters):
                        txtfile.write(
                            f"{global_particle_idx:<12}{area:>20.6f}{diameter:>20.6f}    (Image: {file_info.file_name})\n"
                        )
                        all_areas.append(area)
                        all_diameters.append(diameter)
                        global_particle_idx += 1

                # Write global summary
                txtfile.write("_______________________________\n")
                txtfile.write(
                f"{'Total count':<12}{'Mean Area [' + target_unit + '²]':>20}{'Mean Diameter [' + target_unit + ']':>20}"
                f"{'Max Area [' + target_unit + '²]':>20}{'Max Diameter [' + target_unit + ']':>20}"
                f"{'Min Area [' + target_unit + '²]':>20}{'Min Diameter [' + target_unit + ']':>20}\n"
                )
                txtfile.write(
                    f"{len(all_areas):<12}"
                    f"{np.mean(all_areas):>20.6f}{np.mean(all_diameters):>20.6f}"
                    f"{np.max(all_areas):>20.6f}{np.max(all_diameters):>20.6f}"
                    f"{np.min(all_areas):>20.6f}{np.min(all_diameters):>20.6f}\n"
                )
        except Exception as e:
            print("Error in writing combined statistics to txt file: ", e)

    def _write_header(self, txtfile, unit):
        from datetime import datetime
        txtfile.write("##############################\n")
        txtfile.write("Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        txtfile.write("##############################\n")
        txtfile.write(f"{'Particle No.':<12}{'Area [' + unit + '²]':>20}{'Diameter [' + unit + ']':>20}\n")

    def _write_particle_data(self, txtfile, scaled_areas, scaled_diameters, particle_count, unit):
        import numpy as np
        for label_idx in range(1, particle_count + 1):
            area = scaled_areas[label_idx - 1]
            diameter = scaled_diameters[label_idx - 1]
            txtfile.write(f"{label_idx:<12}{area:>20.6f}{diameter:>20.6f}\n")

        txtfile.write("_______________________________\n")
        txtfile.write(
            f"{'Total count':<12}{'Mean Area [' + unit + '²]':>20}{'Mean Diameter [' + unit + ']':>20}"
            f"{'Max Area [' + unit + '²]':>20}{'Max Diameter [' + unit + ']':>20}"
            f"{'Min Area [' + unit + '²]':>20}{'Min Diameter [' + unit + ']':>20}\n"
        )
        txtfile.write(
            f"{particle_count:<12}"
            f"{np.mean(scaled_areas):>20.6f}{np.mean(scaled_diameters):>20.6f}"
            f"{np.max(scaled_areas):>20.6f}{np.max(scaled_diameters):>20.6f}"
            f"{np.min(scaled_areas):>20.6f}{np.min(scaled_diameters):>20.6f}\n"
        )
