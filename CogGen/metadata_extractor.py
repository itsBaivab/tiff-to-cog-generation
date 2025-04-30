# -*- coding: utf-8 -*-
import time
from datetime import datetime
import logging
import os
import json
from osgeo import gdal
import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver # Or Observer
from watchdog.observers import Observer
from pathlib import Path
import re
import numpy as np

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set base level to INFO

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- Helper Functions ---
def extractTimeStamp(filepath):
    """Extracts timestamp from filename pattern DDMONYYYY_HHMM"""
    pattern = r'(\d{2})([A-Za-z]{3})(\d{4})_(\d{4})'
    basename = os.path.basename(filepath)
    match = re.search(pattern, basename)
    try:
        day, month_str, year, time_str = match.groups()
        month = datetime.strptime(month_str, "%b").month
        timestamp = f"{day.zfill(2)}{str(month).zfill(2)}{year}{time_str}"
        return timestamp
    except AttributeError:
        logger.error(f"Could not extract timestamp pattern from filename: {basename}")
        return None
    except Exception as e:
        logger.error(f"Error parsing timestamp from file {basename}: {str(e)}")
        # Decide if error should be raised or handled with None
        return None # Return None on parsing error

def wait_for_file_stability(filepath, timeout=30, check_interval=0.5):
    """Waits for file size to stabilize before processing."""
    start_time = time.time()
    last_size = -1
    stable_count = 0
    logger.info(f"Waiting for file to stabilize: {filepath}")
    while time.time() - start_time < timeout:
        try:
            if not os.path.exists(filepath):
                time.sleep(check_interval)
                continue
            current_size = os.path.getsize(filepath)
            if (current_size == last_size and current_size >= 0):
                stable_count += 1
                if stable_count >= 3:
                    if current_size > 0:
                        logger.info(f"File size stabilized at {current_size} bytes after {time.time() - start_time:.2f} seconds")
                        return True
                    else:
                         logger.warning(f"File stabilized at 0 bytes: {filepath}. Assuming stable but empty.")
                         return True # Treat 0-byte stable as stable
            else:
                stable_count = 0
            last_size = current_size
            time.sleep(check_interval)
        except FileNotFoundError:
            # File might be deleted during check
            logger.warning(f"File not found during stability check: {filepath}. Retrying check.")
            time.sleep(check_interval)
            last_size = -1; stable_count = 0 # Reset state
        except OSError as e:
            logger.warning(f"OS Error checking file size for {filepath}: {str(e)}. Retrying check.")
            time.sleep(check_interval)
            last_size = -1; stable_count = 0 # Reset state

    logger.warning(f"Timeout waiting for file to stabilize: {filepath}")
    # Final check after timeout
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.warning(f"File {filepath} exists after timeout, proceeding cautiously.")
            return True
    except OSError: pass
    return False

def get_band_stats_from_gdal(band_info):
    """Extract band statistics from GDAL band info"""
    min_val = max_val = mean_val = std_val = None
    
    # Check for statistics in band metadata
    if 'metadata' in band_info and '' in band_info['metadata']:
        metadata = band_info['metadata']['']
        
        # Try to get stats from metadata
        if 'STATISTICS_MINIMUM' in metadata and 'STATISTICS_MAXIMUM' in metadata:
            min_val = float(metadata.get('STATISTICS_MINIMUM'))
            max_val = float(metadata.get('STATISTICS_MAXIMUM'))
            mean_val = float(metadata.get('STATISTICS_MEAN', 0))
            std_val = float(metadata.get('STATISTICS_STDDEV', 0))
        elif 'MIN' in metadata and 'MAX' in metadata:
            min_val = float(metadata.get('MIN'))
            max_val = float(metadata.get('MAX'))
            mean_val = float(metadata.get('MEAN', 0))
            std_val = float(metadata.get('STD_DEV', 0))
    
    # If stats not available in metadata, check if they're in the band info directly
    if min_val is None and 'min' in band_info and 'max' in band_info:
        min_val = float(band_info.get('min'))
        max_val = float(band_info.get('max'))
        mean_val = float(band_info.get('mean', 0))
        std_val = float(band_info.get('stdDev', 0))
    
    logger.info(f"Band stats from GDAL: min={min_val}, max={max_val}, mean={mean_val}, stdDev={std_val}")
    return min_val, max_val, mean_val, std_val

# --- Simplified Image Type and Product Code Logic ---
def extract_image_type_and_product(filename, PRODUCT_CODES, gdal_info=None):
    """Extract image type and product code from filename using consistent logic"""
    filename_parts = filename.split("_")
    image_type = "UNKNOWN"
    product_code = "NONE"
    
    # Check for L2C pattern (higher priority)
    if "_L2C_" in filename:
        # Find the part after L2C
        for i, part in enumerate(filename_parts):
            if part == "L2C" and i+1 < len(filename_parts):
                image_type = filename_parts[i+1]  # Get product code (INS, CMP, etc.)
                product_code = image_type  # For L2C, use image_type as product_code
                break
    
    # Check for L2B pattern
    elif "_L2B_" in filename:
        # Find the part after L2B
        for i, part in enumerate(filename_parts):
            if part == "L2B" and i+1 < len(filename_parts):
                image_type = filename_parts[i+1]  # Get product code (IMC, etc.)
                product_code = image_type  # For L2B, use image_type as product_code
                break
    
    # Check for L2G pattern
    elif "_L2G_" in filename:
        # Find the part after L2G
        for i, part in enumerate(filename_parts):
            if part == "L2G" and i+1 < len(filename_parts):
                image_type = filename_parts[i+1]  # Get product code
                product_code = image_type  # For L2G, use image_type as product_code
                break
    
    # Check for L2P pattern
    elif "_L2P_" in filename:
        # Find the part after L2P
        for i, part in enumerate(filename_parts):
            if part == "L2P" and i+1 < len(filename_parts):
                image_type = filename_parts[i+1]  # Get product code
                product_code = image_type  # For L2P, use image_type as product_code
                break
                
    # Check for multi-band images
    elif gdal_info and 'bands' in gdal_info and len(gdal_info['bands']) > 1:
        image_type = "MULTI"
    
    # Check for IMG specific pattern
    elif "IMG" in filename_parts:
        try:
            img_index = filename_parts.index("IMG")
            if img_index < len(filename_parts) - 1: 
                image_type = filename_parts[img_index + 1]
        except ValueError:
            pass
    
    # If still no product code, try to match against known product codes
    if product_code == "NONE":
        for code in PRODUCT_CODES:
            pattern = f"_{code}_|_{code}\\.|^{code}_|{code}$"
            if re.search(pattern, filename.replace('.cog.tif', '')):
                product_code = code
                # If we still don't have an image type, use the product code
                if image_type == "UNKNOWN":
                    image_type = code
                break
    
    logger.info(f"Extracted: image_type={image_type}, product_code={product_code}")
    return image_type, product_code

# --- TiffHandler Class (Watchdog Event Handler) ---
class TiffHandler(FileSystemEventHandler):
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.base_dir = os.path.abspath(os.getcwd())

    def on_created(self, event):
        """Handles file creation events."""
        filepath = event.src_path
        if event.is_directory or not filepath.lower().endswith('.cog.tif'):
            # Ignore directories and non-COG TIFF files
            # Log ignored files only if necessary, otherwise return silently
            # logger.debug(f"Ignoring event (directory or wrong extension): {filepath}")
            return

        abs_path = os.path.abspath(filepath)
        logger.info(f"New COG TIFF file detected: {abs_path}")

        try:
            if not wait_for_file_stability(abs_path):
                logger.error(f"File did not stabilize or is potentially incomplete: {abs_path}")
                return

            if not os.path.exists(abs_path):
                # File might have been deleted between stabilization and processing
                logger.error(f"File disappeared after stability check: {abs_path}")
                return

            # Optional small delay
            time.sleep(0.5)

            # Extract metadata with retry logic for the whole process
            metadata = self.get_raster_metadata_with_retry(abs_path)

            if metadata:
                 logger.info(f"Metadata extracted for {abs_path}, attempting send.")
                 self.sendmetadata(metadata)
            else:
                 logger.warning(f"Metadata extraction failed for {abs_path}, not sending.")

        except Exception as e:
            # Catch-all for errors during the handling of a single file event
            logger.error(f"Error processing file {filepath}: {str(e)}", exc_info=True)


    def get_raster_metadata_with_retry(self, filepath, max_retries=3, retry_delay=2):
        """Attempts to get metadata, retrying the entire process on failure."""
        last_error = None
        for attempt in range(max_retries):
            try:
                # Ensure file still exists before each attempt
                if not os.path.exists(filepath):
                    logger.error(f"File not found before attempt {attempt+1}: {filepath}")
                    return None

                # Call the main metadata extraction function
                metadata = self.get_raster_metadata(filepath)
                if metadata is not None:
                    logger.info(f"Metadata extraction successful on attempt {attempt + 1} for {filepath}")
                    return metadata # Success
                else:
                    # Case where get_raster_metadata handled an error internally and returned None
                    logger.warning(f"get_raster_metadata returned None on attempt {attempt+1} for {filepath}, retrying...")
                    last_error = ValueError("get_raster_metadata returned None")

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1} failed to get metadata from {filepath}: {str(e)}")
                # No need to log exc_info here unless debugging retries themselves

            # Wait before next attempt (exponential backoff)
            wait_time = retry_delay * (2 ** attempt)
            logger.info(f"Retrying metadata extraction for {filepath} in {wait_time}s...")
            time.sleep(wait_time)

        # Log final failure after all retries
        logger.error(f"All {max_retries} attempts failed for {filepath}: {str(last_error)}", exc_info=True)
        return None


    def get_raster_metadata(self, filepath):
        """Extracts metadata from a single raster file using GDAL."""
        # Define patterns and codes (consider moving PRODUCT_CODES to a config file or constant module)
        version_pattern = r'V(\d{2})'
        revision_pattern = r'R(\d{2})'
        PRODUCT_CODES = [
            'UTH', 'OLR', 'CMK', 'HEM', 'LST', 'SST', 'IMC', 'DHI', 'DNI', 'GHI', 'INS', 'FOG', 'CMP', 'IMR',
            'AOD', 'GPI', 'PET_DLY', 'GHI_DLY', 'DNI_DLY', 'DHI_DLY', 'INS_DLY', 'HEM_DLY',
            'LST_MAX_DLY', 'LST_MIN_DLY', 'UTH_DLY', 'OLR_DLY', 'SST_DLY', 'IMC_DLY', 'SWR_DLY',
            'IMG_TIR1_TEMP', 'IMG_TIR2_TEMP', 'IMG_MIR_TEMP', 'IMG_WV_TEMP', 'IMG_VIS_RADIANCE',
            'IMG_SWIR_RADIANCE', 'IMG_TIR1_RADIANCE', 'IMG_TIR2_RADIANCE', 'IMG_MIR_RADIANCE',
            'IMG_WV_RADIANCE', 'SNW', 'SGP', 'ASIA_MER', 'IMG_SWIR', 'IMG_VIS', 'IMG_TIR1', 'IMG_TIR2', 
            'IMG_MIR', 'IMG_WV', 'CTP', 'WV_MERGED', 'WDP', 'FIR', 'IRW', 'MRW', 'WVW', 'VSW',
        ]

        filename = os.path.basename(filepath)
        logger.info(f"Extracting metadata for: {filename}")
        metadata = None # Initialize result

        try:
            # Open dataset with GDAL
            logger.info(f"Opening file with GDAL: {filepath}")
            ds = gdal.Open(filepath)
            if ds is None:
                logger.error(f"Failed to open the file with GDAL: {filepath}")
                return None
                
            # Get detailed GDAL info as JSON
            logger.info(f"Retrieving GDAL info for: {filepath}")
            gdal_info_str = gdal.Info(ds, deserialize=False, format='json')
            gdal_info = json.loads(gdal_info_str)
            
            logger.info(f"File opened. Bands={gdal_info.get('bands', [])}, Size={gdal_info.get('size', [])}")
            
            # Get filename parts
            filename_parts = filename.split("_")
            file_nodata = None
            if 'bands' in gdal_info and len(gdal_info['bands']) > 0:
                if 'noDataValue' in gdal_info['bands'][0]:
                    file_nodata = gdal_info['bands'][0]['noDataValue']
            
            if file_nodata is not None:
                logger.info(f"File {filename} uses nodata value: {file_nodata}")
            else:
                logger.warning(f"No nodata value defined in metadata for {filename}.")

            # --- Image Type and Product Code Logic ---
            image_type, product_code = extract_image_type_and_product(filename, PRODUCT_CODES, gdal_info)
            logger.info(f"Determined image_type={image_type}, product_code={product_code}")
            # --- End Image Type / Product Code ---

            # Process bands
            bands_info = []
            band_count = len(gdal_info.get('bands', []))
            logger.info(f"Processing {band_count} bands...")
            
            for band_idx, band_info in enumerate(gdal_info.get('bands', []), 1):
                logger.info(f"--- Processing Band {band_idx}/{band_count} ---")
                
                # Get band type
                band_data_type = band_info.get('type', 'Unknown')
                
                # Extract band statistics
                min_val, max_val, mean_val, std_val = get_band_stats_from_gdal(band_info)
                
                # Get band description
                band_description = "unknown"
                if 'description' in band_info:
                    band_description = band_info['description']
                elif band_count == 1:
                    band_description = image_type if image_type != "UNKNOWN" else product_code
                
                # Add band info to our collection
                bands_info.append({
                    "bandId": band_idx, 
                    "data_type": band_data_type,
                    "min": min_val, 
                    "max": max_val, 
                    "minimum": min_val, 
                    "maximum": max_val,
                    "mean": mean_val, 
                    "stdDev": std_val,
                    "nodata_value": band_info.get('noDataValue', file_nodata),
                    "description": band_description
                })
                
                logger.info(f"--- Finished Processing Band {band_idx}/{band_count} ---")
            # End of band loop

            logger.info("Finished processing all bands. Assembling final metadata.")

            # --- Assemble Metadata Dictionary Safely ---
            sat_match = "UNKNOWN"; processingLevel = "UNKNOWN"; version = None; revision = None; productId = "UNKNOWN"; aquisition_datetime = None
            try: sat_match = filename_parts[0].split("IMG")[0] if filename_parts else "UNKNOWN"
            except IndexError: logger.warning(f"Could not parse satellite from {filename}")
            try: processingLevel = filename_parts[3] if len(filename_parts) > 3 else "UNKNOWN"
            except IndexError: logger.warning(f"Could not parse processing level from {filename}")
            try: aquisition_datetime = extractTimeStamp(filename)
            except Exception as ts_e: logger.error(f"Timestamp extraction failed: {ts_e}")
            try: productId = filename.split(".")[0] if '.' in filename else filename
            except Exception: logger.warning(f"Could not parse product ID from {filename}")
            try:
                last_part = productId.split("_")[-1]
                v_match = re.search(version_pattern, last_part)
                r_match = re.search(revision_pattern, last_part)
                if v_match: version = v_match.group(1)
                if r_match: revision = r_match.group(1)
            except Exception as vr_e: logger.warning(f"Could not parse version/revision from {filename}: {vr_e}")

            # Get corner coordinates
            corner_coords = gdal_info.get('cornerCoordinates', {})
            bounds = {
                'left': corner_coords.get('upperLeft', [0, 0])[0],
                'right': corner_coords.get('lowerRight', [0, 0])[0],
                'top': corner_coords.get('upperLeft', [0, 0])[1],
                'bottom': corner_coords.get('lowerRight', [0, 0])[1]
            }
            
            # Assemble metadata structure
            metadata = {
                "filename": filename, 
                "description": gdal_info.get('description', 'unknown'), 
                "satellite": sat_match, 
                "processingLevel": processingLevel,
                "version": version, 
                "revision": revision, 
                "productId": productId,
                "filepath": os.path.abspath(os.path.dirname(filepath)), 
                "aquisition_datetime": aquisition_datetime,
                "type": image_type, 
                "product_code": product_code,
                "coverage": {
                    "lat1": float(bounds['bottom']), 
                    "lat2": float(bounds['top']), 
                    "lon1": float(bounds['left']), 
                    "lon2": float(bounds['right'])
                },
                "coordinateSystem": gdal_info.get('coordinateSystem', {}).get('wkt', None) if 'coordinateSystem' in gdal_info else None,
                "size": {"width": gdal_info.get('size', [0, 0])[0], "height": gdal_info.get('size', [0, 0])[1]},
                "cornerCoords": {
                    "upperLeft": corner_coords.get('upperLeft', [0, 0]),
                    "lowerLeft": corner_coords.get('lowerLeft', [0, 0]),
                    "lowerRight": corner_coords.get('lowerRight', [0, 0]),
                    "upperRight": corner_coords.get('upperRight', [0, 0]),
                    "center": corner_coords.get('center', [0, 0])
                },
                "bands": bands_info
            }
            # --- End Assemble Metadata ---

            # Clean up GDAL resources
            ds = None

        except Exception as e:
            # Catch-all for unexpected errors during metadata assembly
            logger.error(f"Unexpected error getting metadata from file {filepath}: {str(e)}", exc_info=True)
            metadata = None
            raise # Re-raise to trigger retry logic

        logger.info(f"--- Finished get_raster_metadata for: {filepath} ---")
        return metadata


    def sendmetadata(self, metadata):
        """Sends extracted metadata to the specified API endpoint."""
        if not metadata:
             logger.warning("Skipping sending metadata because it's empty.")
             return
        try:
            logger.info(f"Sending metadata for {metadata.get('filename', 'Unknown File')} to {self.api_endpoint}")
            response = requests.post(
                self.api_endpoint,
                json=metadata,
                headers={'Content-Type': 'application/json'},
                timeout=60 # Timeout for the request
            )
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
            logger.info(f"Sent metadata successfully for {metadata.get('filename', 'Unknown File')} (Status: {response.status_code})")
        except requests.exceptions.Timeout:
             logger.error(f"Timeout error sending metadata for {metadata.get('filename', 'Unknown File')} to {self.api_endpoint}")
        except requests.exceptions.ConnectionError as ce:
             logger.error(f"Connection error sending metadata for {metadata.get('filename', 'Unknown File')} to {self.api_endpoint}: {ce}")
        except requests.exceptions.HTTPError as http_err:
             logger.error(f"HTTP error sending metadata for {metadata.get('filename', 'Unknown File')}: {http_err.response.status_code} - {http_err.response.text}")
        except requests.exceptions.RequestException as e:
            # Catch other potential requests errors
            logger.error(f"Error sending metadata for {metadata.get('filename', 'Unknown File')}: {str(e)}")
        except Exception as e:
             # Catch unexpected errors during the send process
             logger.error(f"Unexpected error during metadata sending for {metadata.get('filename', 'Unknown File')}: {str(e)}", exc_info=True)


# --- Directory Watching Logic ---
def watch_directory(path, api_endpoint):
    """Sets up and starts the directory watcher."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        try:
            os.makedirs(abs_path, exist_ok=True)
            logger.info(f"Created directory: {abs_path}")
        except OSError as e:
             logger.error(f"Failed to create directory {abs_path}: {e}")
             return # Cannot watch if directory creation fails

    event_handler = TiffHandler(api_endpoint)
    # Use Observer for efficiency, fall back to PollingObserver if needed (e.g., network drives)
    observer = Observer()
    # observer = PollingObserver()
    observer.schedule(event_handler, abs_path, recursive=True)

    try:
        observer.start()
        logger.info(f"Started watching directory: {abs_path}")
        logger.info(f"Will send metadata to: {api_endpoint}")
        while True:
            # Keep main thread alive while observer runs in background
            time.sleep(5)
            if not observer.is_alive():
                 logger.error("Observer thread died unexpectedly. Stopping script.")
                 break
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping directory watch...")
    except Exception as e:
        # Catch errors in the main watch loop itself (e.g., observer start issues)
        logger.error(f"Error in main watch loop: {e}", exc_info=True)
    finally:
        # Cleanly stop the observer
        if observer.is_alive():
            observer.stop()
            logger.info("Observer stopped.")
        observer.join() # Wait for observer thread to finish
        logger.info("Observer joined. Exiting.")


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Metadata Extractor Script Started.")
    # Get config from environment variables or use defaults
    WATCH_DIR = os.environ.get("WATCH_DIR", "/home/sbn/baivab/final") # Adjust default as needed
    API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://localhost:7000/api/metadata/save") # Adjust default as needed

    logger.info(f"Watching directory: {WATCH_DIR}")
    logger.info(f"Sending metadata to: {API_ENDPOINT}")

    # Optional: Set logger level via environment variable for easier debugging
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    logger.info(f"Logging level set to: {log_level_str}")
    # To enable DEBUG logging, set environment variable: export LOG_LEVEL=DEBUG

    # Configure GDAL
    gdal.UseExceptions()  # Enable exceptions from GDAL
    logger.info("GDAL configured to use exceptions")

    # Validate API endpoint format
    if not API_ENDPOINT.startswith(("http://", "https://")):
        logger.error(f"Invalid API_ENDPOINT format: {API_ENDPOINT}. Should start with http:// or https://")
    else:
        watch_directory(WATCH_DIR, API_ENDPOINT)

    logger.info("Metadata Extractor Script Finished.")
