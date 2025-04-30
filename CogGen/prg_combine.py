import time
from datetime import datetime
import logging
import os
import rasterio
import requests
import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver
from watchdog.observers import Observer
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


def extractTimeStamp(filepath):
    pattern = r'(\d{2})([A-Za-z]{3})(\d{4})_(\d{4})'
    match = re.search(pattern, os.path.basename(filepath))
    try:
        day, month_str, year, time = match.groups()
        month = datetime.strptime(month_str, "%b").month
        return f"{day.zfill(2)}{str(month).zfill(2)}{year}{time}"
    except Exception as e:
        logger.error(f"Error getting metadata from file: {str(e)}")
        raise


def wait_for_file_stability(filepath, timeout=30, check_interval=0.5):
    """Wait for file to be completely written by checking if its size stabilizes"""
    start_time = time.time()
    last_size = -1
    stable_count = 0
    
    logger.info(f"Waiting for file to stabilize: {filepath}")
    
    while time.time() - start_time < timeout:
        try:
            current_size = os.path.getsize(filepath)
            
            # If size hasn't changed from last check
            if (current_size == last_size and current_size > 0):
                stable_count += 1
                # Consider file stable after 3 consecutive unchanged size checks
                if stable_count >= 3:
                    logger.info(f"File size stabilized at {current_size} bytes after {time.time() - start_time:.2f} seconds")
                    return True
            else:
                # Reset stable count if size changed
                stable_count = 0
                
            last_size = current_size
            time.sleep(check_interval)
            
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Error checking file size: {str(e)}")
            time.sleep(check_interval)
    
    logger.warning(f"Timeout waiting for file to stabilize: {filepath}")
    return False


class TiffHandler(FileSystemEventHandler):
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.base_dir = os.path.abspath(os.getcwd())

    def on_created(self, event):
        """Handle file creation events"""
        try:
            if (event.is_directory or not event.src_path.lower().endswith('.cog.tif')):
                return
                
            # Get absolute path
            abs_path = os.path.abspath(event.src_path)
            logger.info(f"New COG TIFF file detected: {abs_path}")

            # Wait for the file to be completely written
            if not wait_for_file_stability(abs_path):
                logger.error(f"File may not be completely written: {abs_path}")
                return
                
            if not os.path.exists(abs_path):
                logger.error(f"File not found: {abs_path}")
                return
                
            # Add a small additional delay for safety
            time.sleep(0.5)
            
            # Process the file
            metadata = self.get_raster_metadata(abs_path)
            if metadata:
                logger.info(f"Metadata extracted for {abs_path}, attempting send.")
                self.sendmetadata(metadata)
            else:
                logger.error(f"Failed to extract metadata for {abs_path}")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)

    def get_raster_metadata(self, filepath):
        """Extract comprehensive metadata from TIFF file with enhanced band value logging"""
        version = r'V(\d{2})'
        revision = r'R(\d{2})'

        # Define all valid product codes
        PRODUCT_CODES = [
            'UTH', 'OLR', 
            'CMK', 'HEM', 'LST', 'SST', 'IMC', 'DHI', 'DNI', 'GHI', 'INS', 'FOG', 'CMP', 'IMR', 
            'AOD', 'GPI', 'PET_DLY', 'GHI_DLY', 'DNI_DLY', 'DHI_DLY', 'INS_DLY', 'HEM_DLY', 
            'LST_MAX_DLY', 'LST_MIN_DLY', 'UTH_DLY', 'OLR_DLY', 'SST_DLY', 'IMC_DLY', 'SWR_DLY', 
            'IMG_TIR1_TEMP', 'IMG_TIR2_TEMP', 'IMG_MIR_TEMP', 'IMG_WV_TEMP', 'IMG_VIS_RADIANCE', 
            'IMG_SWIR_RADIANCE', 'IMG_TIR1_RADIANCE', 'IMG_TIR2_RADIANCE', 'IMG_MIR_RADIANCE', 
            'IMG_WV_RADIANCE', 'SNW','SGP','ASIA_MER',
        ]

        filename = os.path.basename(filepath)
        logger.info(f"Extracting metadata for: {filename}")
        
        try:
            with rasterio.open(filepath) as src:
                logger.info(f"File opened. Bands={src.count}, Dtypes={src.dtypes}, Shape={src.shape}")
                filename_parts = filename.split("_")
                file_nodata = src.nodata
                
                if file_nodata is not None:
                    logger.info(f"File {filename} uses nodata value: {file_nodata}")
                else:
                    logger.warning(f"No nodata value defined in metadata for {filename}.")
                
                # Check if this is an L1B file (which needs masked arrays)
                is_l1b_file = "_L1B_" in filename
                
                # Determine if we should use masking based on file type and nodata presence
                should_use_masking = is_l1b_file or file_nodata is not None
                logger.info(f"File type detection: is_l1b_file={is_l1b_file}")
                
                # Extract product code for L2B and L2C files
                image_type = "UNKNOWN"  # Default value
                
                # Check for L2C pattern
                if "_L2C_" in filename:
                    for i, part in enumerate(filename_parts):
                        if part == "L2C" and i+1 < len(filename_parts):
                            image_type = filename_parts[i+1]
                            break
                
                # Check for L2B pattern
                elif "_L2B_" in filename:
                    for i, part in enumerate(filename_parts):
                        if part == "L2B" and i+1 < len(filename_parts):
                            image_type = filename_parts[i+1]
                            break
                
                # Check for L1B pattern
                elif "_L1B_" in filename:
                    for i, part in enumerate(filename_parts):
                        if part == "L1B" and i+1 < len(filename_parts):
                            image_type = filename_parts[i+1]
                            break
                    if image_type == "UNKNOWN":
                        image_type = "L1B"
                
                # If not L2B/L2C/L1B or couldn't extract product code, use the existing logic
                elif src.count > 1:
                    image_type = "MULTI"
                else:
                    if "IMG" in filename_parts:
                        try:
                            img_index = filename_parts.index("IMG")
                            if img_index < len(filename_parts) - 1:
                                image_type = filename_parts[img_index + 1]
                        except ValueError:
                            pass
                
                # Find any product code in the filename
                product_code = "NONE"

                # First check for codes in L2C and L2B files (most specific)
                if "_L2C_" in filename or "_L2B_" in filename or "_L1B_" in filename:
                    if image_type != "UNKNOWN":
                        product_code = image_type
                else:
                    # For other file types, check for each product code in the list
                    for code in PRODUCT_CODES:
                        pattern = f"_{code}_|_{code}\\.|^{code}_"
                        if re.search(pattern, filename):
                            product_code = code
                            break

                # If still NONE, try one more method - check if any code appears anywhere
                if product_code == "NONE":
                    for code in PRODUCT_CODES:
                        if code in filename:
                            product_code = code
                            break

                logger.info(f"Determined image_type={image_type}, product_code={product_code}")
                
                # Check for FOG product which might need scaling
                is_fog_product = product_code == "FOG"
                scale_factor = 1/128 if is_fog_product else None
                if scale_factor:
                    logger.info(f"Using scale factor {scale_factor} for {product_code} product")
                
                # Process bands
                bands_info = []
                logger.info(f"Processing {src.count} bands with{'' if should_use_masking else 'out'} masking...")
                
                for band_idx in range(1, src.count + 1):
                    logger.info(f"Processing band {band_idx}/{src.count}...")
                    
                    try:
                        if should_use_masking:
                            logger.info(f"Reading band {band_idx} with masking")
                            band = src.read(band_idx, masked=True)
                            
                            if isinstance(band, np.ma.MaskedArray):
                                valid_pixel_count = band.count()
                                logger.info(f"Band {band_idx} has {valid_pixel_count} valid pixels")
                                
                                if valid_pixel_count > 0:
                                    min_val = float(band.min())
                                    max_val = float(band.max())
                                    mean_val = float(band.mean(dtype=np.float64))
                                    std_val = float(band.std(dtype=np.float64))
                                    
                                    logger.info(f"Raw band {band_idx} stats: min={min_val}, max={max_val}, mean={mean_val:.6f}, std={std_val:.6f}")
                                    
                                    # Apply scale factor if needed
                                    if scale_factor:
                                        min_val *= scale_factor
                                        max_val *= scale_factor
                                        mean_val *= scale_factor
                                        std_val *= scale_factor
                                        logger.info(f"Scaled band {band_idx} stats: min={min_val}, max={max_val}, mean={mean_val:.6f}, std={std_val:.6f}")
                                else:
                                    logger.warning(f"Band {band_idx} has no valid pixels")
                                    min_val = max_val = mean_val = std_val = None
                            else:
                                logger.warning(f"Expected masked array but got {type(band)}")
                                min_val = float(np.min(band))
                                max_val = float(np.max(band))
                                mean_val = float(np.mean(band, dtype=np.float64))
                                std_val = float(np.std(band, dtype=np.float64))
                                logger.info(f"Band {band_idx} stats: min={min_val}, max={max_val}, mean={mean_val:.6f}, std={std_val:.6f}")
                        else:
                            logger.info(f"Reading band {band_idx} without masking (standard approach)")
                            band = src.read(band_idx)
                            min_val = float(band.min())
                            max_val = float(band.max())
                            mean_val = float(band.mean())
                            std_val = float(band.std())
                            logger.info(f"Band {band_idx} stats: min={min_val}, max={max_val}, mean={mean_val:.6f}, std={std_val:.6f}")
                            
                            # Apply scale factor if needed
                            if scale_factor:
                                min_val *= scale_factor
                                max_val *= scale_factor
                                mean_val *= scale_factor
                                std_val *= scale_factor
                                logger.info(f"Scaled band {band_idx} stats: min={min_val}, max={max_val}, mean={mean_val:.6f}, std={std_val:.6f}")
                        
                        # Set band description
                        if src.count == 1:
                            band_description = image_type
                        else:
                            band_description = "unknown"
                            try:
                                if src.descriptions and len(src.descriptions) >= band_idx:
                                    if src.descriptions[band_idx-1]:
                                        band_description = src.descriptions[band_idx-1]
                            except Exception as desc_e:
                                logger.warning(f"Could not read band description: {desc_e}")
                        
                        bands_info.append({
                            "bandId": band_idx,
                            "data_type": str(band.dtype),
                            "min": min_val,
                            "max": max_val,
                            "minimum": min_val,
                            "maximum": max_val,
                            "mean": mean_val,
                            "stdDev": std_val,
                            "nodata_value": file_nodata,
                            "description": band_description
                        })
                        
                        logger.info(f"---- Finished Processing Band {band_idx}/{src.count} ---")
                        
                    except Exception as band_e:
                        logger.error(f"Error processing band {band_idx}: {str(band_e)}", exc_info=True)
                
                logger.info("Finished processing all bands. Assembling final metadata.")
                
                # Assemble metadata
                metadata = {
                    "filename": filename,
                    "description": 'unknown',
                    "satellite": filename.split("_")[0].split("IMG")[0],
                    "processingLevel": filename.split("_")[3],
                    "version": re.search(version, str(filename.split("_")[-1])).group(1),
                    "revision": re.search(revision, str(filename.split("_")[-1])).group(1),
                    "productId": filename.split(".")[0],
                    "filepath": os.path.dirname(filepath),
                    "aquisition_datetime": extractTimeStamp(filename),
                    "type": image_type,
                    "product_code": product_code,
                    "coverage": {
                        "lat1": float(src.bounds.bottom),
                        "lat2": float(src.bounds.top),
                        "lon1": float(src.bounds.left),
                        "lon2": float(src.bounds.right)
                    },
                    "coordinateSystem": src.crs.to_string(),
                    "size": {
                        "width": src.width,
                        "height": src.height
                    },
                    "cornerCoords": {
                        "upperLeft": [src.bounds.left, src.bounds.top],
                        "lowerLeft": [src.bounds.left, src.bounds.bottom],
                        "lowerRight": [src.bounds.right, src.bounds.bottom],
                        "upperRight": [src.bounds.right, src.bounds.top],
                        "center": [
                            (src.bounds.left + src.bounds.right) / 2,
                            (src.bounds.top + src.bounds.bottom) / 2
                        ]
                    },
                    "bands": bands_info
                }
                
                logger.info(f"---- Finished get_raster_metadata for: {filepath} ---")
                return metadata
                
        except Exception as e:
            logger.error(f"Error getting metadata from file {filepath}: {str(e)}", exc_info=True)
            return None

    def sendmetadata(self, metadata):
        """Send metadata to API endpoint"""
        try:
            logger.info(f"Sending metadata for {metadata['filename']} to {self.api_endpoint}")
            response = requests.post(
                self.api_endpoint,
                json=metadata,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info(f"Sent metadata successfully for {metadata['filename']} (Status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending metadata: {str(e)}")


def watch_directory(path, api_endpoint):
    """Set up directory watching"""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
        logger.info(f"created directory:{abs_path}")
    event_handler = TiffHandler(api_endpoint)
    observer = PollingObserver()
    observer.schedule(event_handler, abs_path, recursive=True)
    observer.start()
    logger.info(f"Started watching directory: {abs_path}")
    logger.info(f"Will send metadata to: {api_endpoint}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopping directory watch")
    observer.join()


if __name__ == "__main__":
    logger.info("Metadata Extractor Script Started.")
    logger.info(f"Watching directory: /home/sbn/baivab/final")
    logger.info(f"Sending metadata to: http://localhost:7000/api/metadata/save")
    logger.info(f"Logging level set to: {logging.getLevelName(logger.level)}")
    
    # WATCH_DIR = "/home/sbn/baivab/cog-testing/metadata-extractor/COG/final_dir"
    WATCH_DIR = "/home/sbn/baivab/final"
    API_ENDPOINT = "http://localhost:7000/api/metadata/save"
    # API_ENDPOINT = "http://localhost:4040/post-json"
    watch_directory(WATCH_DIR, API_ENDPOINT)
