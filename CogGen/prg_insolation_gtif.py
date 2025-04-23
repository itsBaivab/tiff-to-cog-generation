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

osr.UseExceptions()

logs_dir = './LOGS'

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

class EventHandler(FileSystemEventHandler):

    #product_code = ['DHI','DNI','GHI','INS','GHI_DLY','DNI_DLY','DHI_DLY','INS_DLY','HEM_DLY','AOD']
    Months       = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    #opath_3r = {'DHI':['T2','T2S1','T2S1P1'],'DNI':['T2','T2S1','T2S1P2'],'GHI':['T2','T2S1','T2S1P3'],'INS':['T2','T2S1','T2S1P4'],'DNI_DLY':['T2','T2S1','insat3r_dni_daily_T2S1P10'],'DHI_DLY':['T2','T2S1','insat3r_dhi_daily_T2S1P11'],'INS_DLY':['T2','T2S1','insat3r_ins_daily_T2S1P6'],'GHI_DLY':['T2','T2S1','insat3r_ghi_daily_T2S1P7'],'HEM_DLY':['T5','T5S3','insat_hem_daily_T5S3P5'],'LST':['T5','T5S2','land_surface_temperature_insat3r_T5S2P3'],'AOD':['T5','T5S1','aod_T5S1P8']}
    #opath_3d = {'DNI_DLY':['T2S1','insat3d_dni_daily_T2S1P9'],'DHI_DLY':['T2S1','insat3d_dhi_daily_T2S1P12'],'INS_DLY':['T2S1','insat3d_ins_daily_T2S1P5'],'GHI_DLY':['T2S1','insat3d_ghi_daily_T2S1P8']}
    #opath_3s ={'DHI':['T2','T2S1','DHI_3S_T2S1P13'],'DNI':['T2','T2S1','DNI_3S_T2S1P14'],'GHI':['T2','T2S1','GHI_3S_T2S1P15'],
#'INS':['T2','T2S1','INS_3S_T2S1P16'],'DNI_DLY':['T2','T2S1','DNI_DLY_3S_T2S1P18'],'DHI_DLY':['T2','T2S1','DHI_DLY_3S_T2S1P19'],'INS_DLY':['T2','T2S1','INS_DLY_3S_T2S1P21'],'GHI_DLY':['T2','T2S1','GHI_DLY_3S_T2S1P17'],'HEM_DLY':['T5','T5S3','HEM_DLY_3S_T2S1P20'],'LST':['T5','T5S2','LST_3S_T2S1P23'],'AOD':['T5','T5S1','AOD_3S_T2S1P22']}
    #lpp_code = ['L1C','L2C','L3C','L2B','L3B','L2G']
    #sat_code = ['3RIMG','3DIMG','3SIMG']

    def __init__(self,outputdir,product_code,lpp_code,sat_code):
        self.outputdir = outputdir 
        self.product_code = product_code
        self.lpp_code = lpp_code
        self.sat_code = sat_code

    def create_directory(self,filename,scode):
        pattern = r'_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_'     #_29MAR2024_0000_
        pattern2 = r'^([^_]+_){4}'                          #3DIMG_29MAR2024_0000_L3B_
        pattern3 = r'^([^_]+){1}'                           #3DIMG
        pattern4 = r'V(\d{2})R(\d{2})'                      #V01R00

        mtch = re.search(pattern,filename)
        mtch_2 = re.search(pattern2,filename)
        mtch_3 = re.search(pattern3,filename)
        mtch_4 = re.search(pattern4,filename)
        
        if 'L1C' in filename:
            sector_name = re.search(r'L1C_(.*?)_V',filename).group(1)

        dirpath = None
        if mtch :
            DD,MMM,YYYY,HHMM = mtch.groups()	
            dt = datetime.strptime(str.format('%s%s%s%s'%(DD,MMM,YYYY,HHMM)),'%d%b%Y%H%M')
            
            if mtch_3.group(0) in self.sat_code:
                dirpath = os.path.join(self.outputdir,mtch_3.group(0),str(YYYY),'%.2d'%(self.Months[MMM]),'%.2d'%(int(DD)))

            os.makedirs(dirpath,exist_ok=True)
            tmstr = str.format('%s%.2d%.2d%sP%s'%(YYYY,self.Months[MMM],int(DD),HHMM,HHMM))
            tfilename = f"{mtch_2.group(0)}{scode}_{mtch_4.group(0)}.tif"
            if 'L1C' in filename:
                tfilename = f"{mtch_2.group(0)}{sector_name}_{scode}_{mtch_4.group(0)}.tif"
						
        return dirpath,tfilename

    def on_created(self, event):
        print(f"on_created: {str(event.src_path)}")
        write_log('COGGEN', f"Processing new file: {str(event.src_path)}", 'INFO')
        fl_basename = os.path.basename(str(event.src_path))
        fl, ext = os.path.splitext(fl_basename)

        if ext != '.h5':
            write_log('COGGEN', f"Skipping non-h5 file: {fl_basename}", 'INFO')
            return
        
        write_log('COGGEN', f"Detected h5 file: {fl_basename}", 'INFO')
        
        # Wait for file to be fully written
        last_size = -1
        wait_attempts = 0
        write_log('COGGEN', f"Waiting for file stability: {fl_basename}", 'INFO')
        
        while True:
            try:
                csize = os.path.getsize(str(event.src_path))
                write_log('COGGEN', f"Current file size: {csize} bytes", 'DEBUG')
                
                if csize == last_size:
                    write_log('COGGEN', f"File size stabilized at {csize} bytes after {wait_attempts} attempts", 'INFO')
                    break
                    
                last_size = csize
                wait_attempts += 1
                time.sleep(2)
            except Exception as e:
                write_log('COGGEN', f"Error checking file size: {str(e)}", 'ERROR')
                return

        # Get datasets from the file
        write_log('COGGEN', f"Getting datasets for file: {fl_basename}", 'INFO')
        datasets = self.getdatasetlist(str(event.src_path))
        write_log('COGGEN', f"Found datasets: {datasets}", 'INFO')
        write_log('COGGEN', f"Looking for product codes {self.product_code} in datasets {datasets}", 'INFO')

        if len(datasets) == 0:
            write_log('COGGEN', f"No datasets found in file: {fl_basename}", 'WARNING')
            return
            
        plevel = self.getprocessinglevel(str(event.src_path))
        write_log('COGGEN', f"Processing level identified: {plevel}", 'INFO')
        
        # Process each product code
        products_processed = 0
        
        for code in self.product_code:
            if code in datasets:
                write_log('COGGEN', f"Processing product code {code} in file: {fl_basename}", 'INFO')
                dirpath, tfilename = self.create_directory(fl, code)
                write_log('COGGEN', f"Target directory: {dirpath}", 'INFO')
                write_log('COGGEN', f"Target filename: {tfilename}", 'INFO')
                
                tfl = os.path.join(dirpath, tfilename)
                
                try:
                    # Process based on processing level
                    if plevel in ['L1B', 'L2B', 'L3B']:
                        write_log('COGGEN', f"Using create_geotif for {plevel} file: {fl_basename}, product: {code}", 'INFO')
                        create_geotif(str(event.src_path), code, tfl)
                        write_log('COGGEN', f'Successfully written: {tfl}', 'INFO')
                        products_processed += 1
                    elif plevel in ['L2G', 'L3G']:
                        write_log('COGGEN', f"Using create_gridded_geotif for {plevel} file: {fl_basename}, product: {code}", 'INFO')
                        create_gridded_geotif(str(event.src_path), code, tfl)
                        write_log('COGGEN', f'Successfully written: {tfl}', 'INFO')
                        products_processed += 1
                    else:
                        # This should handle L1C, L2C, L3C
                        write_log('COGGEN', f"Using write_toa_geotiff for {plevel} file: {fl_basename}, product: {code}", 'INFO')
                        write_toa_geotiff(str(event.src_path), code, tfl)
                        write_log('COGGEN', f'Successfully written: {tfl}', 'INFO')
                        products_processed += 1
                        
                    # Check if COG conversion is needed for tfl
                    if os.path.exists(tfl):
                        # Comment out the COG creation in prg_insolation_gtif.py
                        # cog_filename = f"{tfl}.cog.tif"
                        # write_log('COGGEN', f"Creating COG file: {cog_filename}", 'INFO')
                        # create_cog(tfl, cog_filename)
                        # write_log('COGGEN', f"Successfully created COG file: {cog_filename}", 'INFO')
                        
                        # No need to remove original file since we're not creating COGs here
                        pass
                    else:
                        write_log('COGGEN', f"Output file not found for COG conversion: {tfl}", 'WARNING')
                        
                except Exception as e:
                    write_log('COGGEN', f"Error processing product {code} in file {fl_basename}: {str(e)}", 'ERROR')
                    print(f"Error processing product {code}: {str(e)}")
        
        write_log('COGGEN', f"Completed processing file {fl_basename}, {products_processed} products processed", 'INFO')
        if products_processed == 0:
            write_log('COGGEN', f"WARNING: No products were processed for file {fl_basename}", 'WARNING')

    def on_moved(self, event):

        fl_basename = os.path.basename(str(event.src_path))
        fl, ext = os.path.splitext(fl_basename)
        datasets = self.getdatasetlist(str(event.src_path))

        for code in self.product_code:
            if code in datasets:
                dirpath,tfilename = self.create_directory(fl,code)
                tfl = os.path.join(dirpath,tfilename)
                write_log('COGGEN',f"Watchdog received moved event - {event.src_path}", level='INFO')
                write_toa_geotiff(str(event.src_path),code,tfl)
                write_log('COGGEN',f'Successfully written: {tfl}','INFO')

    def getdatasetlist(self,filename):
        bfilename = os.path.basename(filename)
        datasets = []
        for code in self.lpp_code:
            if code in bfilename:
                ds = h5.File(filename,'r')
                datasets = list(ds.keys())
                ds.close()
        return datasets

    def getprocessinglevel(self,filename):
        #bfilename = os.path.basename(filename)
        file_attrs = {}
        ds = h5.File(filename,'r')
        file_attrs.update(ds.attrs)
        ds.close()
        return file_attrs['Processing_Level'].astype(str)

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

def get_corner_cordinates(l1c_hdf,dataset,geotiff_filename:str):

    write_log('COGGEN',f'reading {l1c_hdf} dataset: {dataset}','INFO')
    proj_param = {}
    band_attrs = {}
    band = None
    NoDataValue = None
    geotiff_filename_upd = None
    try:
        dst_h5 = h5.File(l1c_hdf,'r')
        if 'L1C' in l1c_hdf and (dataset.find('_TEMP') > 0 or dataset.find('_RADIANCE')> 0):
            band,NoDataValue,geotiff_filename_upd = apply_lookup_table(dst_h5,dataset,geotiff_filename)
            band_attrs['_FillValue'] = np.asarray([NoDataValue])
        else:
            bnd = dst_h5.get(dataset)
            band_attrs.update(bnd.attrs)
            band    = np.squeeze(bnd[()])

        dst_proj = dst_h5.get('Projection_Information')
        dst_proj = dst_proj.attrs
        proj_param.update(dst_proj)
        dst_h5.close() 
    except Exception as e:
        write_log('COGGEN',f'Exception in opening {l1c_hdf}-{str(e)}','ERROR')
        dst_h5.close()

    return proj_param,band_attrs,band,geotiff_filename_upd

def conver_lonlat_to_xy(proj_param,step=0):
    	
    clat = proj_param['standard_parallel'][0]
    clon = proj_param['longitude_of_projection_origin'][0]
    fe   = proj_param['false_easting'][0]
    fn   = proj_param['false_northing'][0]
    grid_mapping_name = proj_param['grid_mapping_name']
    if step == 0:
        lat,lon  = proj_param['upper_left_lat_lon(degrees)']
    elif step == 1:
        lat,lon  = proj_param['lower_right_lat_lon(degrees)']
        write_log('COGGEN',f"Lower right corner: Latitude {lat}, Longitude {lon}", level='INFO')

    write_log('COGGEN',f"Central Latitude: {clat}, Central Longitude: {clon}", level='INFO')
    bmap = pyproj.Proj(proj="merc",lat_ts=clat,lon_0=clon,ellps="WGS84")
    crs = pyproj.CRS.from_proj4(bmap.to_proj4())

    return  bmap(lon,lat,inverse=False),crs.to_wkt()

def write_toa_geotiff(l3c_hdf_filename,dataset,l3c_geotiff_filename):

    proj_param,band_attrs,band,geotiff_filename_upd = get_corner_cordinates(l3c_hdf_filename,dataset,l3c_geotiff_filename)
    if geotiff_filename_upd != None:
        l3c_geotiff_filename = geotiff_filename_upd
    dtype = band.dtype.name 

    if dtype == 'uint16':
        GDT_Type = gdal.GDT_UInt16
    elif dtype == 'float32':
        GDT_Type = gdal.GDT_Float32
    else:
        GDT_Type = gdal.GDT_Float32

    metadata = get_band_stats(band,int(band_attrs['_FillValue'][0]))

    print(f'----{len(proj_param.keys())} {len(band_attrs.keys())} {dtype} {GDT_Type}')
    if len(proj_param.keys()) == 0 or len(band_attrs.keys()) == 0:
        write_log('COGGEN',f'file {l3c_hdf_filename} either corrupted or {dataset} missing','ERROR')
        return

    dst_ul_xy,wkt = conver_lonlat_to_xy(proj_param)
    dst_lr_xy,wkt = conver_lonlat_to_xy(proj_param,1)

    write_log('COGGEN',f"Upper-left corner (x, y): {dst_ul_xy[0]}, {dst_ul_xy[1]}", level='INFO')
    write_log('COGGEN',f"Lower-right corner (x, y): {dst_lr_xy[0]}, {dst_lr_xy[1]}", level='INFO')
    write_log('COGGEN',f"Band shape: {band.shape}", level='INFO')	

    we_pixel_resolution = dst_lr_xy[0] - dst_ul_xy[0]
    ns_pixel_resolution  = dst_ul_xy[1] - dst_lr_xy[1]

    nscan,npix = band.shape

    xres = we_pixel_resolution/npix
    yres = -ns_pixel_resolution/nscan

    write_log('COGGEN',f"xres: {xres}, yres: {yres}", level='INFO')
    write_log('COGGEN',f"nscan: {nscan}, npix: {npix}", level='INFO')

    drv = gdal.GetDriverByName( 'GTiff' )
    dst_ds = drv.Create(l3c_geotiff_filename, npix,nscan,1, GDT_Type )
    dst_ds.SetProjection(wkt)
    dst_ds.SetGeoTransform([dst_ul_xy[0],xres,0.0,dst_ul_xy[1],0.0,yres])
    b1 = dst_ds.GetRasterBand(1)
    b1.SetMetadata(metadata)
    b1.SetStatistics(np.float64(metadata['MIN']),np.float64(metadata['MAX']),np.float64(metadata['MEAN']),np.float64(metadata['STD_DEV']))
    b1.SetNoDataValue(int(band_attrs['_FillValue'][0]))
    b1.WriteArray(band)
    dst_ds = None

def create_cog(inputfile,outputfile):
    gdal.Translate(outputfile,inputfile,format="COG",options=["COMPRESS=LZW","BIGTIFF=IF_SAFER"])


def apply_lookup_table(h5_fid,dataset:str,geotiff_filename:str):

    lut_list = ['IMG_TIR1_TEMP','IMG_TIR2_TEMP','IMG_MIR_TEMP','IMG_WV_TEMP','IMG_VIS_RADIANCE','IMG_SWIR_RADIANCE','IMG_TIR1_RADIANCE','IMG_TIR2_RADIANCE','IMG_MIR_RADIANCE','IMG_WV_RADIANCE']
    
    lut_dict = {'IMG_TIR1_TEMP':'TIR1_TEMP','IMG_TIR2_TEMP':'TIR2_TEMP','IMG_MIR_TEMP':'MIR_TEMP','IMG_WV_TEMP':'WV_TEMP','IMG_VIS_RADIANCE':'VIS_RADIANCE','IMG_SWIR_RADIANCE':'SWIR_RADIANCE','IMG_TIR1_RADIANCE':'TIR1_RADIANCE','IMG_TIR2_RADIANCE':'TIR2_RADIANCE','IMG_MIR_RADIANCE':'MIR_RADIANCE','IMG_WV_RADIANCE':'WV_RADIANCE'}

    if dataset in lut_list and 'TEMP' in dataset:
        cnt_dataset = dataset.replace('_TEMP','').strip()
    else :
        cnt_dataset = dataset.replace('_RADIANCE','').strip()

    geotiff_filename = geotiff_filename.replace(dataset,lut_dict[dataset])

    lut = h5_fid.get(dataset)
    lut_attrs = lut.attrs
    lut_nodatavalue = lut_attrs['_FillValue'][0]
    lut = np.squeeze(lut[()])

    cnt = h5_fid.get(cnt_dataset)
    cnt_attrs = cnt.attrs
    cnt_nodatavalue = cnt_attrs['_FillValue'][0]
    cnt = np.squeeze(cnt[()])

    phyq_nodatavalue = -999.0
    phyq = lut[cnt]
    phyq[cnt == cnt_nodatavalue] = phyq_nodatavalue

    return phyq,phyq_nodatavalue,geotiff_filename



def create_geotif(hdf_filename,dataset,geotiff_filename):
    gdal.SetConfigOption('CPL_NUM_THREADS','ALL_CPUS')
    inpfl,ext = os.path.splitext(os.path.basename(hdf_filename))


    file = h5.File(hdf_filename)
    if 'L1B' in hdf_filename and (dataset.find('_TEMP') > 0 or dataset.find('_RADIANCE')> 0):
        dset,NoDataValue,gtiff_filename_upd = apply_lookup_table(file,dataset,geotiff_filename)
        geotiff_filename = gtiff_filename_upd
        #print(f"{geotiff_filename}")
    else:
        dset = file.get(dataset)
        dset_attr = dset.attrs
        NoDataValue = dset_attr['_FillValue'][0]
        dset = np.squeeze(dset[()])

    if dataset in ['IMG_SWIR','IMG_VIS','IMG_VIS_RADIANCE','IMG_SWIR_RADIANCE']:
        dset = dset[227:11037,187:11033]
        xRes = 0.00909 
        yRes = 0.00909
    elif dataset in ['IMG_TIR1','IMG_TIR2','IMG_MIR','IMG_TIR1_TEMP','IMG_TIR2_TEMP','IMG_MIR_TEMP','IMG_TIR1_RADIANCE','IMG_TIR2_RADIANCE','IMG_MIR_RADIANCE']:
        dset = dset[57:2759,47:2757]
        xRes=0.03636
        yRes=0.03636
    elif dataset in ['IMG_WV','IMG_WV_TEMP','IMG_WV_RADIANCE']:
        dset = dset[29:1379,24:1378]
        xRes =0.07272
        yRes= 0.07272
    else:
        dset = dset[57:2759,47:2757]
        xRes=0.03636
        yRes=0.03636

    file.close()

    dtype = dset.dtype.name

    if dtype == 'uint16':
        GDT_Type = gdal.GDT_UInt16
    elif dtype == 'float32':
        GDT_Type = gdal.GDT_Float32
    else:
        GDT_Type = gdal.GDT_Float32

    #print(f'----{dtype} {GDT_Type}')
    metadata = get_band_stats(dset,NoDataValue)

    chain = inpfl[0:5]
    globe = ccrs.Globe(semimajor_axis=6378137.0,semiminor_axis=6356752.31414)
    if chain == '3DIMG' or chain == '3SIMG':
        proj = ccrs.Geostationary(central_longitude=82.0,satellite_height=35785831,false_easting=0,false_northing=0,sweep_axis='x',globe=globe)
    if chain == '3RIMG' :
        proj = ccrs.Geostationary(central_longitude=74.0,satellite_height=35785831,false_easting=0,false_northing=0,sweep_axis='x',globe=globe)

    geos = osr.SpatialReference()
    geos.SetWellKnownGeogCS('WGS84')
    geos.SetProjCS("GEOS")
    geos.ImportFromProj4(proj.proj4_init)
    wkt = geos.ExportToWkt()

    drv = gdal.GetDriverByName('MEM')

    dst_ds = drv.Create("", dset.shape[1],dset.shape[0],1, GDT_Type )
    dst_ds.SetProjection(wkt)
    xres = (proj.x_limits[1] - proj.x_limits[0])/dset.shape[1]
    yres = (proj.y_limits[1] - proj.y_limits[0])/dset.shape[0]


    dst_ds.SetGeoTransform([proj.x_limits[1],xres,0.0,proj.y_limits[0],0.0,-yres])
    b1 = dst_ds.GetRasterBand(1)
    b1.SetMetadata(metadata)
    #b1.SetStatistics(np.float64(metadata['MEAN']),np.float64(metadata['STD_DEV']),np.float64(metadata['MIN']),np.float64(metadata['MAX']))
    b1.SetStatistics(np.float64(metadata['MIN']),np.float64(metadata['MAX']),np.float64(metadata['MEAN']),np.float64(metadata['STD_DEV']))
    b1.WriteArray(dset)
    b1.SetNoDataValue(int(NoDataValue))
    
    temp_dir = '/tmp'
    inter_gtif = os.path.join(temp_dir,'%s_OUT.tif'%(inpfl))

    #print(f'{inter_gtif} {geotiff_filename}')

    #options = ['NUM_THREADS=ALL_CPUS']
    creation_options = ["NUM_THREADS=ALL_CPUS"]

    ds = gdal.Translate(inter_gtif, dst_ds, outputBounds = [proj.x_limits[0],proj.y_limits[1],proj.x_limits[1],proj.y_limits[0]],resampleAlg="near",noData=NoDataValue,creationOptions=creation_options)
    #ds = gdal.Translate(inter_gtif, dst_ds, creationOptions=creation_options)
    #print('Translate Over')
    ws = gdal.Warp(geotiff_filename, ds, dstSRS='EPSG:4326',xRes=xRes,yRes=yRes,resampleAlg="near",srcNodata=NoDataValue,dstNodata=NoDataValue,multithread=True)
    ds = None
    ws = None
    file.close()
    os.remove(inter_gtif)
    # cog_filename=f"{geotiff_filename}.cog.tif"
    # create_cog(geotiff_filename,cog_filename)


def create_gridded_geotif(hdf_filename,dataset,geotiff_filename):
    fl = h5.File(hdf_filename)
    fl_attrs = {}
    fl_attrs.update(fl.attrs)
    ds = fl.get(dataset)[()]
    ds = np.squeeze(ds)
    ds_attrs = {}

    ds_attrs.update(fl.get(dataset).attrs)

    fl.close()

    dtype = ds.dtype.name

    if dtype == 'uint16':
        GDT_Type = gdal.GDT_UInt16
    elif dtype == 'float32':
        GDT_Type = gdal.GDT_Float32
    else:
        GDT_Type = gdal.GDT_Float32

    print(f'----{dtype} {GDT_Type}')
    metadata = get_band_stats(ds,int(ds_attrs['_FillValue'][0]))

    drv = gdal.GetDriverByName( 'GTiff' )

    dst_ds = drv.Create(geotiff_filename, ds.shape[1],ds.shape[0],1, GDT_Type )


    dst_ds.SetProjection('EPSG:4326')
    dst_ds.SetGeoTransform([fl_attrs['left_longitude'][0],fl_attrs['lon_interval'][0],0.0,fl_attrs['upper_latitude'][0],0.0,-fl_attrs['lat_interval'][0]])
    b1 = dst_ds.GetRasterBand(1)
    b1.SetMetadata(metadata)
    #b1.SetStatistics(np.float64(metadata['MEAN']),np.float64(metadata['STD_DEV']),np.float64(metadata['MIN']),np.float64(metadata['MAX']))
    b1.SetStatistics(np.float64(metadata['MIN']),np.float64(metadata['MAX']),np.float64(metadata['MEAN']),np.float64(metadata['STD_DEV']))
    b1.SetNoDataValue(int(ds_attrs['_FillValue'][0]))
    b1.WriteArray(ds)
    dst_ds = None
    
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
    logs_dir = config['COG_GENERATION']['LOGS']

    event_handler = EventHandler(output_dir,product_codes,lpp_codes,sat_codes)
    observer = PollingObserver()
    for watch_dir in watch_dirs:
        observer.schedule(event_handler, watch_dir, recursive=True)
    observer.start()
    print(f"Monitoring of  {watch_dirs} started")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
 
