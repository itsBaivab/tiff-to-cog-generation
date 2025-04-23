import time
from datetime import datetime
import logging
import os
import rasterio
import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from watchdog.observers import Observer
from pathlib import Path
import re


# logger = setup_logging()   # initializing the logger


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
        logger.error(f"Error getting metadata from file :{str(e)}")
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
            logger.info(f"New TIFF file detected: {abs_path}")

            # Wait for the file to be completely written
            if not wait_for_file_stability(abs_path):
                logger.error(f"File may not be completely written: {abs_path}")
                return
                
            if not os.path.exists(abs_path):
                logger.error(f"File not found: {abs_path}")
                return
                
            # Add a small additional delay for safety
            time.sleep(0.5)
            
            # Use retry logic for reading the file
            metadata = self.get_raster_metadata_with_retry(abs_path)
            self.sendmetadata(metadata)
            
        except Exception as e:
            logger.error(f"Error Processing file: {str(e)}", exc_info=True)

    def get_raster_metadata_with_retry(self, filepath, max_retries=3, retry_delay=2):
        """Attempt to get metadata with retries if it fails"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.get_raster_metadata(filepath)
            except Exception as e:
                last_error = e
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Attempt {attempt+1} failed to get metadata. Retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        logger.error(f"All {max_retries} attempts failed: {str(last_error)}")
        raise last_error

    def get_raster_metadata(self, filepath):
        """ Extract comprehensive metadata from TIFF file"""
        pattern = r'_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_'  # _29MAR2024_0000_
        pattern2 = r'^([^_]+_){4}'  # 3DIMG_29MAR2024_0000_L3B_
        pattern3 = r'^([^_]+){1}'  # 3DIMG
        pattern4 = r'V(\d{2})R(\d{2})'  # V01R00
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

        print(os.path.basename(filepath))
        print(os.path.basename(filepath).split("_")[-1])
        try:
            with rasterio.open(filepath)as src:
                # Determine image type based on filename and band count
                filename = os.path.basename(filepath)
                filename_parts = filename.split("_")
                
                # Extract product code for L2B and L2C files
                image_type = "UNKNOWN"  # Default value
                
                # Check for L2C pattern (e.g., 3RIMG_17APR2025_0045_L2C_INS_V01R00)
                if "_L2C_" in filename:
                    # Find the part after L2C
                    for i, part in enumerate(filename_parts):
                        if part == "L2C" and i+1 < len(filename_parts):
                            image_type = filename_parts[i+1]  # Get product code (INS, CMP, etc.)
                            break
                
                # Check for L2B pattern (e.g., 3RIMG_L2B_IMC)
                elif "_L2B_" in filename:
                    # Find the part after L2B
                    for i, part in enumerate(filename_parts):
                        if part == "L2B" and i+1 < len(filename_parts):
                            image_type = filename_parts[i+1]  # Get product code (IMC, etc.)
                            break
                
                # If not L2B/L2C or couldn't extract product code, use the existing logic
                elif src.count > 1:
                    image_type = "MULTI"
                else:
                    # For single band images that aren't L2B/L2C, use existing logic
                    if "IMG" in filename_parts:
                        # Find the band type that comes after IMG
                        img_index = filename_parts.index("IMG")
                        if img_index < len(filename_parts) - 1:
                            image_type = filename_parts[img_index + 1]  # Get band type (VIS, SWIR, TIR2, etc.)
                
                # Find any product code in the filename
                product_code = "NONE"

                # First check for codes in L2C and L2B files (most specific)
                if "_L2C_" in filename or "_L2B_" in filename:
                    # Use the already extracted image_type if it's not UNKNOWN
                    if image_type != "UNKNOWN":
                        product_code = image_type
                else:
                    # For other file types, check for each product code in the list
                    for code in PRODUCT_CODES:
                        # Extract exact matches - look for code surrounded by underscores or at the end of filename
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

                # Add debugging
                print(f"Extracted product_code: {product_code} for file: {filename}")
                
                bands_info = []
                for band_idx in range(1, src.count+1):
                    band = src.read(band_idx)
                    
                    # Set band description
                    if src.count == 1:
                        # For single band files, use the image_type as the description
                        band_description = image_type
                    else:
                        # For multi-band files, use the original logic
                        band_description = "unknown"
                        try:
                            if src.descriptions and len(src.descriptions) >= band_idx:
                                if src.descriptions[band_idx-1]:
                                    band_description = src.descriptions[band_idx-1]
                        except Exception:
                            # Fall back to default if description not available
                            pass
                    
                    bands_info.append({
                        "bandId": band_idx,
                        "data_type": str(band.dtype),
                        "min": float(band.min()),
                        "max": float(band.max()),
                        "minimum": float(band.min()),
                        "maximum": float(band.max()),
                        "mean": float(band.mean()),
                        "stdDev": float(band.std()),
                        "nodata_value": src.nodata,
                        "description": band_description
                    })
                
                metadata = {
                    "filename": os.path.basename(filepath),
                    "description": 'unknown',
                    "satellite": os.path.basename(filepath).split("_")[0].split("IMG")[0],
                    "processingLevel": os.path.basename(filepath).split("_")[3],
                    "version":  re.search(version, str(os.path.basename(filepath)).split("_")[-1]).group(1),
                    "revision": re.search(revision, str(os.path.basename(filepath)).split("_")[-1]).group(1),
                    "productId": os.path.basename(filepath).split(".")[0],
                    "filepath": os.path.dirname(filepath),
                    "aquisition_datetime": extractTimeStamp(os.path.basename(filepath)),
                    "type": image_type,  # The image type field
                    "product_code": product_code,  # New field for product code
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
                    "cornerCoords":{
                        "upperLeft":[src.bounds.left,src.bounds.top],
                        "lowerLeft":[src.bounds.left,src.bounds.bottom],
                        "lowerRight":[src.bounds.right,src.bounds.bottom],
                        "upperRight":[src.bounds.right,src.bounds.top],
                        "center":[
                            (src.bounds.left+src.bounds.right)/2,
                            (src.bounds.top+src.bounds.bottom)/2
                            
                            ]
                        },
                    "bands": bands_info

                }
                return metadata
        except Exception as e:
            logger.error(f"Error getting metadata from file :{str(e)}")
            raise

    def sendmetadata(self, metadata):
        """Send metadata to API endpoint"""
        try:
            response = requests.post(
                self.api_endpoint,
                json=metadata,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info(
                f"Sent metadata successfully for {metadata['filename']}")
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
    # WATCH_DIR = "/home/sbn/baivab/cog-testing/metadata-extractor/COG/final_dir"
    WATCH_DIR = "/home/sbn/baivab/final"
    API_ENDPOINT = "http://localhost:7000/api/metadata/save"
    # API_ENDPOINT = "http://localhost:4040/post-json"
    watch_directory(WATCH_DIR, API_ENDPOINT)
