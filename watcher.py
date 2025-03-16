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
            if event.is_directory or not event.src_path.lower().endswith('.tif'):
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
        """Extract comprehensive metadata from TIFF file"""
        try:
            with rasterio.open(filepath) as src:
                bands_info = []
                for band_idx in range(1, src.count + 1):
                    band = src.read(band_idx)
                    bands_info.append({
                        "band_number": band_idx,
                        "statistics": {
                            "min": float(band.min()),
                            "max": float(band.max()),
                            "mean": float(band.mean()),
                            "std_dev": float(band.std())
                        },
                        "properties": {
                            "nodata_value": src.nodata,
                            "color_interpretation": str(src.colorinterp[band_idx-1]),
                            "data_type": str(band.dtype)
                        }
                    })

                metadata = {
                    "file": {
                        "absolute_path": filepath,
                        "filename": os.path.basename(filepath),
                        "directory": os.path.dirname(filepath),
                        "size_bytes": os.path.getsize(filepath),
                        "creation_time": datetime.fromtimestamp(
                            os.path.getctime(filepath)
                        ).isoformat()
                    },
                    "raster": {
                        "dimensions": {
                            "width": src.width,
                            "height": src.height,
                            "bands": src.count
                        },
                        "spatial_reference": {
                            "crs": src.crs.to_string(),
                            "transform": [float(x) for x in src.transform[:6]],
                            "resolution_meters": {
                                "x": abs(src.transform[0]),
                                "y": abs(src.transform[4])
                            }
                        },
                        "format": {
                            "driver": src.driver,
                            "compression": src.profile.get("compress", "none"),
                            "photometric": src.profile.get("photometric", "none")
                        }
                    },
                    "bands": bands_info
                }
                return metadata
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