import re

# Sample filenames
filenames = [
    "3DIMG_18JAN2024_1030_L1C_ASIA_MER_V01R00.h5",
    "3RIMG_16OCT2024_0115_L1C_ASIA_MER_V01R00_IMG_MIR.tif",
    "3RIMG_16OCT2024_0115_L1C_ASIA_MER_V01R00_IMG_SWIR.tif",
    "3RIMG_16OCT2024_0115_L1C_ASIA_MER_V01R00_IMG_TIR1.tif",
    "3RIMG_16OCT2024_0115_L1C_ASIA_MER_V01R00_IMG_TIR2.tif",
    "3RIMG_16OCT2024_0115_L1C_ASIA_MER_V01R00_IMG_VIS.tif",
    "3RIMG_16OCT2024_0115_L1C_ASIA_MER_V01R00_IMG_WV.tif"
]

# Regex pattern to extract relevant information
pattern = re.compile(r'(?P<satellite>\d[DR]IMG)_(?P<date>\d{2}[A-Z]{3}\d{4})_(?P<time>\d{4})_(?P<level>L1C)_(?P<region>ASIA)_(?P<type>MER)_(?P<version>V\d{2}R\d{2})(?:_IMG_(?P<band>[A-Z0-9]+))?')

# Extracted data storage
data_list = []

for filename in filenames:
    match = pattern.search(filename)
    if match:
        data = match.groupdict()
        data_list.append(data)

# Print extracted data
for item in data_list:
    print(item)
