import time
import json
import rasterio
import requests
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TiffHandler(FileSystemEventHandler):
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.base_dir = os.path.abspath(os.getcwd())

    def on_created(self, event):
        """Handle file creation events"""
        try:
            if event.is_directory or not event.src_path.lower().endswith('.cog.tif'):
                return

            # Get absolute path
            abs_path = os.path.abspath(event.src_path)
            logger.info(f"New TIFF detected: {abs_path}")

            # Wait for file to be completely written
            time.sleep(1)  # Add small delay to ensure file is ready

            if not os.path.exists(abs_path):
                logger.error(f"File not found: {abs_path}")
                return

            metadata = self.get_raster_metadata(abs_path)
            self.send_metadata(metadata)

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)

    def get_raster_metadata(self, filepath):
        """Extract metadata matching target schema"""
        try:
            with rasterio.open(filepath) as src:
                filename = os.path.basename(filepath)
                # Extract information from filename
                match = re.match(r'3RIMG_(\d{2})(\w{3})(\d{4})_(\d{4})_(\w{3})_(\w+)_(\w+)_V(\d{2})R(\d{2})', filename)
                
                if match:
                    day, month, year, time, level, region, product, version, revision = match.groups()
                    date_str = f"{day}{month}{year}"
                    date_obj = datetime.strptime(date_str, '%d%b%Y')
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                    
                    # Get product ID without extension
                    product_id = filename.split('.')[0]
                    if '_IMG_' in product_id:
                        product_id = product_id.split('_IMG_')[0]

                    # Get bounds in geographic coordinates
                    bounds = src.bounds
                    transform = src.transform
                    
                    # Calculate corner coordinates
                    corners = {
                        'upperleft': [bounds.left, bounds.top],
                        'lowerleft': [bounds.left, bounds.bottom],
                        'upperright': [bounds.right, bounds.top],
                        'lowerright': [bounds.right, bounds.bottom],
                        'center': [
                            (bounds.left + bounds.right) / 2,
                            (bounds.bottom + bounds.top) / 2
                        ]
                    }

                    # Process bands
                    bands_info = []
                    for band_idx in range(1, src.count + 1):
                        band = src.read(band_idx)
                        bands_info.append({
                            "bandID": band_idx,
                            "type": str(band.dtype),
                            "colorInterpretation": str(src.colorinterp[band_idx-1].value),
                            "min": float(band.min()),
                            "max": float(band.max()),
                            "mean": float(band.mean()),
                            "std_dev": float(band.std()),
                            "nodata_value": float(src.nodata) if src.nodata is not None else None
                        })

                    metadata = {
                        "productID": product_id,
                        "name": product_id,
                        "description": product_id,
                        "satellite": "INSAT3R",
                        "aquisition_data": formatted_date,
                        "processing_level": level,
                        "version": version,
                        "revision": revision,
                        "filename": filename,
                        "filepath": os.path.dirname(filepath),
                        "coverage": {
                            "lat1": float(bounds.bottom),
                            "lat2": float(bounds.top),
                            "lon1": float(bounds.left),
                            "lon2": float(bounds.right)
                        },
                        "cordinate_system": src.crs.to_string(),
                        "size": {
                            "width": src.width,
                            "height": src.height,
                            "bands": src.count
                        },
                        "coornercoords": corners,
                        "bands": bands_info
                    }
                    return metadata
                else:
                    raise ValueError("Filename does not match expected pattern")

        except Exception as e:
            logger.error(f"Error reading raster file: {str(e)}")
            raise

    def send_metadata(self, metadata):
        """Send metadata to API endpoint"""
        try:
            response = requests.post(
                self.api_endpoint,
                json=metadata,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info(f"Metadata sent successfully for {metadata['file']['filename']}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending metadata: {str(e)}")

def watch_directory(path, api_endpoint):
    """Set up directory watching"""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
        logger.info(f"Created directory: {abs_path}")

    event_handler = TiffHandler(api_endpoint)
    observer = Observer()
    observer.schedule(event_handler, abs_path, recursive=False)
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
    WATCH_DIR = "./final_dir"
    API_ENDPOINT = "http://localhost:5000/post-metadata"
    watch_directory(WATCH_DIR, API_ENDPOINT)