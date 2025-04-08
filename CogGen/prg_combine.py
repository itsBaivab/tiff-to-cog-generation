import numpy as np
import h5py as h5
import sys
import os 
from osgeo import gdal
from osgeo import osr
import pyproj
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import time
import re
from datetime import datetime
#from geos_gtif import create_geotif
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cartopy, cartopy.crs as ccrs
import configparser
import time
import statistics
from collections import defaultdict

osr.UseExceptions()

logs_dir = '/usr/local/logs/CogGen'

def write_log(rule,message,level='INFO'):
    log_file_name = get_dynamic_log_file_name(rule)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f'{timestamp} - {level} - {rule} - {message}\n'
    with open(log_file_name,'a') as log_file:
        log_file.write(log_entry)
    log_file.close()

def get_dynamic_log_file_name(rule='COGGEN'):
    now = datetime.now()
    #print('----',os.getcwd())
    return os.path.join(logs_dir,f'{rule}_log_{now.strftime("%Y%m%d")}.log')

def get_l1b_listing(inp_directory:str):
    pattern_l1b = re.compile(r"^(3RIMG_\d{2}[A-Za-z]{3}\d{4}_\d{4}_L\d{1}[A-Za-z]{1})_([^_]+_[^_]+)_(V\d{2}R\d{2})\.tif$")
    groups_l1b = defaultdict(list)
    files = os.listdir(inp_directory)

    for file in files:
        #print(f"{file}")
        match_l1b = pattern_l1b.match(file)

        if match_l1b:
            prefix = match_l1b.group(1)
            #lpp    = match.group(2)
            version = match_l1b.group(3)
            key = f"{prefix}_{version}"
            #print(f"{prefix}_{version}")
            groups_l1b[key].append(file)

    return groups_l1b

def get_l1c_listing(inp_directory:str):
    pattern_l1c = re.compile(r"^(3RIMG_\d{2}[A-Za-z]{3}\d{4}_\d{4}_L\d{1}[A-Za-z]{1})_([^_]+)_([^_]+_[^_]+)_(V\d{2}R\d{2})\.tif$")
    groups_l1c = defaultdict(list)
    files = os.listdir(inp_directory)

    for file in files:
        match_l1c = pattern_l1c.match(file)
        #print(f"{file} {match_l1c}")
        if match_l1c:
            prefix = match_l1c.group(1)
            sector    = match_l1c.group(2)
            version = match_l1c.group(4)
            key = f"{prefix}_{sector}_{version}"
            groups_l1c[key].append(file)


    return groups_l1c

def get_l1c_asia_listing(inp_directory:str):
    pattern_l1c = re.compile(r"^(3RIMG_\d{2}[A-Za-z]{3}\d{4}_\d{4}_L\d{1}[A-Za-z]{1})_([^_]+_[^_]+)_([^_]+_[^_]+)_(V\d{2}R\d{2})\.tif$")
    groups_l1c = defaultdict(list)
    files = os.listdir(inp_directory)

    for file in files:
        match_l1c = pattern_l1c.match(file)
        #print(f"{file} {match_l1c}")
        if match_l1c:
            prefix = match_l1c.group(1)
            sector    = match_l1c.group(2)
            version = match_l1c.group(4)
            key = f"{prefix}_{sector}_{version}"
            groups_l1c[key].append(file)


    return groups_l1c

class EventHandler(FileSystemEventHandler):

    Months       = {'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06','JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12'}

    def __init__(self,outputdir,product_code,lpp_code,sat_code,merge_scripts):
        self.outputdir = outputdir
        self.product_code = product_code
        self.lpp_code = lpp_code
        self.sat_code = sat_code
        self.merge_script = merge_scripts

    def parse_insat_filename(self,fl):
        pattern = r'^([^_]+){1}'
        satsen = re.search(pattern,fl)
        pattern = r"_(\d{2})([A-Za-z]{3})(\d{4})_"
        date = re.search(pattern,fl)
        #print(f"{satsen} {date}")
        return date,satsen

    def create_directory(self,filename,scode):
        print(f"dummy")
        #return dirpath,tfilename

    def create_cog(self,input_gtif:str,output_cog:str):
        create_options = ["COMPRESS=LZW","BIGTIFF=IF_SAFER","RESAMPLING=NEAREST"]
        src_ds = gdal.Open(input_gtif,gdal.GA_ReadOnly)
        
        # Get the band description before translation
        band_desc = None
        if src_ds.RasterCount > 0:
            band = src_ds.GetRasterBand(1)
            band_desc = band.GetDescription()
        
        # Create the COG
        dst_ds = gdal.Translate(output_cog, src_ds, format='COG', creationOptions=create_options)
        
        # Set the band description in the output file if one was found
        if band_desc and dst_ds:
            out_band = dst_ds.GetRasterBand(1)
            out_band.SetDescription(band_desc)
            dst_ds = None
        
        src_ds = None

    def resample_and_merge(self,input_files:list,output_file:str):
        predict_seq = ['IMG_VIS','IMG_SWIR','IMG_TIR1','IMG_TIR2','IMG_MIR','IMG_WV','VIS_RADIANCE','SWIR_RADIANCE','TIR1_RADIANCE','TIR2_RADIANCE','MIR_RADIANCE','WV_RADIANCE','TIR1_TEMP','TIR2_TEMP','MIR_TEMP','WV_TEMP']

        print("Merging Process Started") 
        target_nodata = -999.0

        src = gdal.Open(input_files[2],gdal.GA_ReadOnly)
        cols = src.RasterXSize
        rows = src.RasterYSize
        geotransform = src.GetGeoTransform()
        projection = src.GetProjection()


        driver = gdal.GetDriverByName('GTiff')
        outputs = driver.Create(output_file,cols,rows,len(input_files),gdal.GDT_Float32)
        outputs.SetGeoTransform(geotransform)
        outputs.SetProjection(projection)


        for idx,file in enumerate(input_files):
            cog_gtif_file = f"{os.path.splitext(file)[0]}.cog.tif"            
            src_ds = gdal.Open(file,gdal.GA_ReadOnly)
            xcols = src_ds.RasterXSize
            yrows = src_ds.RasterYSize

            if cols != xcols or rows != yrows:

                metadata = src_ds.GetRasterBand(1).GetMetadata()
                nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()
                
                dst_ds = gdal.GetDriverByName('MEM').Create("",cols,rows,1,gdal.GDT_Float32)
                dst_ds.SetGeoTransform(geotransform)
                dst_ds.SetProjection(projection)

                #ws = gdal.Warp(dst_ds, src_ds,resampleAlg="near",srcNodata=nodata_value,dstNodata=nodata_value,multithread=True)
                gdal.ReprojectImage(src_ds, dst_ds)


                band = dst_ds.GetRasterBand(1).ReadAsArray()
                #if 'IMG_VIS' in file or 'IMG_SWIR' in file:
                print(f"source nodata value: {nodata_value}")
                #band[band == nodata_value] = target_nodata
                band[band == 0] = target_nodata

                b1 = outputs.GetRasterBand(idx+1)
                b1.SetMetadata(metadata)
                b1.SetStatistics(np.float64(metadata['MIN']),np.float64(metadata['MAX']),np.float64(metadata['MEAN']),np.float64(metadata['STD_DEV'])) 
                #b1.SetNoDataValue(nodata_value)
                b1.SetNoDataValue(target_nodata)
                b1.WriteArray(band)
                b1.SetDescription(predict_seq[idx])
                src_ds = None
                ouputs = None
                dst_ds = None
            else:
                metadata = src_ds.GetRasterBand(1).GetMetadata()
                nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()

                band = src_ds.GetRasterBand(1).ReadAsArray()
                #if 'IMG_VIS' in file or 'IMG_SWIR' in file:
                #    band[band == 0] = 1023
                band= band.astype(np.float32)
                band[band == nodata_value] = target_nodata
                b1 = outputs.GetRasterBand(idx+1)
                b1.SetMetadata(metadata)
                b1.SetStatistics(np.float64(metadata['MIN']),np.float64(metadata['MAX']),np.float64(metadata['MEAN']),np.float64(metadata['STD_DEV'])) 
                #b1.SetNoDataValue(nodata_value)
                b1.SetNoDataValue(target_nodata)
                b1.WriteArray(band)
                b1.SetDescription(predict_seq[idx])
                src_ds = None
            
            self.create_cog(file,cog_gtif_file)
            os.remove(file)
        ouputs = None
        print("Merging Process Completed") 

    def get_band_stats(band,fill_value=None):

        if fill_value is not None:
            mask = np.isin(band,fill_value)
            valid_values = band[~mask]
        else:
            valid_values = band[~np.isnan(band)]

        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        mean_val = np.mean(valid_values)
        std_dev = np.std(valid_values)
        mode_val = statistics.mode(valid_values.flatten())
        metadata = {"MIN": str(min_val),"MAX": str(max_val),"MEAN": str(mean_val),"STD_DEV":str(std_dev),"MODE":str(mode_val)}
        return metadata

    def wait_for_file_stability(self, filepath, timeout=30, check_interval=0.5):
        """Wait for file to be completely written by checking if its size stabilizes"""
        start_time = time.time()
        last_size = -1
        stable_count = 0
        
        print(f"Waiting for file to stabilize: {filepath}")
        
        while time.time() - start_time < timeout:
            try:
                current_size = os.path.getsize(filepath)
                
                # If size hasn't changed from last check
                if (current_size == last_size and current_size > 0):
                    stable_count += 1
                    # Consider file stable after 3 consecutive unchanged size checks
                    if stable_count >= 3:
                        print(f"File size stabilized at {current_size} bytes after {time.time() - start_time:.2f} seconds")
                        return True
                else:
                    # Reset stable count if size changed
                    stable_count = 0
                    
                last_size = current_size
                time.sleep(check_interval)
                
            except (FileNotFoundError, OSError) as e:
                print(f"Error checking file size: {str(e)}")
                time.sleep(check_interval)
        
        print(f"Timeout waiting for file to stabilize: {filepath}")
        return False

    def on_created(self, event):
        predict_seq = ['IMG_VIS','IMG_SWIR','IMG_TIR1','IMG_TIR2','IMG_MIR','IMG_WV','VIS_RADIANCE','SWIR_RADIANCE','TIR1_RADIANCE','TIR2_RADIANCE','MIR_RADIANCE','WV_RADIANCE','TIR1_TEMP','TIR2_TEMP','MIR_TEMP','WV_TEMP']
        print(f"on_created: {str(event.src_path)}")
        fl_basename = os.path.basename(str(event.src_path))
        fl, ext = os.path.splitext(fl_basename)

        if ext != '.tif' or '.cog' in fl:
            return
        
        # Fix 1: Check if file exists before trying to access it
        if not os.path.exists(event.src_path):
            print(f"File not found immediately after creation event: {event.src_path}")
            return
            
        try:
            # Fix 2: Use proper wait_for_file_stability method (fix self reference)
            if not self.wait_for_file_stability(event.src_path, timeout=60, check_interval=1):
                print(f"File may not be completely written: {event.src_path}")
                return
                
            # Now it's safe to continue with processing
            if '_L1B_' in fl:
                #print(f"Eureka")
                mtch,satsen = self.parse_insat_filename(fl)
                directory = os.path.join(self.outputdir,satsen.group(1),mtch.group(3),self.Months[mtch.group(2)],mtch.group(1))
                groups_l1b = get_l1b_listing(directory)
                for key,group in groups_l1b.items():
                    if (len(group)) == 16 and 'WV_RADIANCE' in fl :
                        abs_file = [os.path.join(directory,file) for file in group]
                        abs_file = rearrange = sorted(abs_file, key = lambda x: predict_seq.index(next(filter(lambda y: y in x, predict_seq),' ')))
                        merge_gtif_file = f"{directory}{os.sep}{key}.tif"
                        cog_gtif_file = f"{directory}{os.sep}{key}.cog.tif"
                        merge_cmd = f'python {self.merge_script} -separate -o {merge_gtif_file} {" ".join(abs_file)}'
                        self.resample_and_merge(abs_file,merge_gtif_file)
                        self.create_cog(merge_gtif_file,cog_gtif_file)
                        #os.remove(merge_gtif_file)
                    
            elif '_L1C_' in fl:

                mtch,satsen = self.parse_insat_filename(fl)
                directory = os.path.join(self.outputdir,satsen.group(1),mtch.group(3),self.Months[mtch.group(2)],mtch.group(1))

                if 'ASIA_MER' in fl:
                    groups_l1c = get_l1c_asia_listing(directory)
                else:
                    groups_l1c = get_l1c_listing(directory)

                for key,group in groups_l1c.items():
                    if (len(group)) == 16 and 'WV_RADIANCE' in fl:
                        abs_file = [os.path.join(directory,file) for file in group]
                        abs_file = rearrange = sorted(abs_file, key = lambda x: predict_seq.index(next(filter(lambda y: y in x, predict_seq),' ')))
                        merge_gtif_file = f"{directory}{os.sep}{key}.tif"
                        cog_gtif_file = f"{directory}{os.sep}{key}.cog.tif"
                        merge_cmd = f'python {self.merge_script} -separate -o {merge_gtif_file} {" ".join(abs_file)}'
                        self.resample_and_merge(abs_file,merge_gtif_file)
                        self.create_cog(merge_gtif_file,cog_gtif_file)
                        #os.remove(merge_gtif_file)

            elif '_L2C_' in fl or '_L2B_' in fl or '_L3C_' in fl or '_L3B_' in fl:
                input_filename = str(event.src_path)
                flname,ext = os.path.splitext(input_filename)
                cog_gtif_file = '%s.cog%s'%(flname,ext)
                self.create_cog(input_filename,cog_gtif_file)
                #os.remove(input_filename)

        except Exception as e:
            print(f"Error processing file {event.src_path}: {str(e)}")
            # You might want to log this error more formally

    #def on_moved(self, event):

    #    fl_basename = os.path.basename(str(event.src_path))
    #    fl, ext = os.path.splitext(fl_basename)
    #    datasets = self.getdatasetlist(str(event.src_path))



def main():

    if (len(sys.argv) < 2):
        print(f"Usage {sys.argv[0]} <config.ini>")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    watch_dirs = config['COG_GENERATION']['WATCH_DIR'].split(',')  # All but the last argument are input directories
    output_dir = config['COG_GENERATION']['OUTPUT_DIR']  # The last argument is the output directory
    sat_codes  = config['COG_GENERATION']['SAT_CODES'].split(',')
    lpp_codes   = config['COG_GENERATION']['LPP_CODES'].split(',')
    product_codes = config['COG_GENERATION']['PRODUCT_CODES'].split(',')
    merge_scripts = config['COG_GENERATION']['MERGE_GTIF']
    logs_dir = config['COG_GENERATION']['LOGS']

    event_handler = EventHandler(output_dir,product_codes,lpp_codes,sat_codes,merge_scripts)
    observer = PollingObserver()
    #for watch_dir in output_dir:
    observer.schedule(event_handler, output_dir, recursive=True)
    observer.start()
    print(f"Monitoring of  {output_dir} started")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

main()
