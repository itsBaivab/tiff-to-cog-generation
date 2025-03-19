import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

def reproject_raster(src_path, dst_crs, resolution=4000):
    """
    Reproject a raster to specified CRS and resolution
    resolution in meters (4000 = 4km)
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, 
            src.width, src.height, 
            resolution=resolution,
            *src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        data = np.empty((height, width), dtype=kwargs['dtype'])
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        return data, kwargs

def combine_cogs(input_files, output_file):
    """
    Combine multiple COG files into a single multi-band COG
    """
    # Use the first file to get the target CRS
    with rasterio.open(input_files[0]) as first:
        dst_crs = first.crs
    
    # Reproject and collect all bands
    all_data = []
    for file in input_files:
        data, kwargs = reproject_raster(file, dst_crs)
        all_data.append(data)
    
    # Update profile for multi-band output
    kwargs.update({
        'count': len(input_files),
        'driver': 'COG',
        'compress': 'LZW',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'overview_resampling': 'average'
    })
    
    # Write combined COG
    with rasterio.open(output_file, 'w', **kwargs) as dst:
        for idx, data in enumerate(all_data, 1):
            dst.write(data, idx)
        
        # Build overviews
        factors = [2, 4, 8, 16]
        dst.build_overviews(factors, Resampling.average)

if __name__ == "__main__":
    # List your input COG files
    input_files = [
        './raw-cog/3RIMG_16OCT2024_0815_L1C_ASIA_MER_V01R00_IMG_VIS.tif',
        './raw-cog/3RIMG_16OCT2024_0815_L1C_ASIA_MER_V01R00_IMG_MIR.tif',
        './raw-cog/3RIMG_16OCT2024_0815_L1C_ASIA_MER_V01R00_IMG_SWIR.tif',
        './raw-cog/3RIMG_16OCT2024_0815_L1C_ASIA_MER_V01R00_IMG_TIR1.tif',
        './raw-cog/3RIMG_16OCT2024_0815_L1C_ASIA_MER_V01R00_IMG_TIR2.tif',
        './raw-cog/3RIMG_16OCT2024_0815_L1C_ASIA_MER_V01R00_IMG_WV.tif'


    ]
    
    output_file = 'final_dir/3RIMG_16OCT2024_0815_L1C_ASIA_MER_V01R00.cog.tif'
    combine_cogs(input_files, output_file)