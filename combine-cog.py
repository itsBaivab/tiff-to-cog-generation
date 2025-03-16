foo = raster.open('file1.tif')
foo_data = foo.read(1)

bar = raster.open('file2.tif')
bar_data = bar.read(1)

tiff_data = [foo_data, bar_data]


with rasterio.open(
    "dummy.tif",
    **{**foo.profile, "count": 2},
    mode="w",
) as file:
    for band, data in enumerate(tiff_data, start=1):
        file.write(data, band)