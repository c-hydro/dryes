import gdal
import numpy as np

# filename= ('/Users/lauro/Documents/PROJECTS/BOLIVIA/Bolivia2/simulazioni/Prova6/Results6b_verbose2_WATERDEPTH_36000.tif');

# readFile(filename):
"""def readGeotiff(filename):
    filehandle = gdal.Open(filename)
    band1 = filehandle.GetRasterBand(1)
    scale = band1.GetScale()
    print (scale)
    geotransform = filehandle.GetGeoTransform()
    geoproj = filehandle.GetProjection()
    band1data = band1.ReadAsArray()
    xsize = filehandle.RasterXSize
    ysize = filehandle.RasterYSize
    return band1data, xsize, ysize, geotransform, geoproj
"""

def readGeotiff (filename):
    filehandle = gdal.Open(filename)
    geotransform = filehandle.GetGeoTransform()
    geoproj = filehandle.GetProjection()
    xsize = filehandle.RasterXSize
    ysize = filehandle.RasterYSize
    band_tot = filehandle.RasterCount

    if band_tot > 1:
        data3d= np.zeros((ysize,xsize,band_tot))

    for i in range (0,band_tot):
        band = filehandle.GetRasterBand(i+1)
        scale = band.GetScale()
        nodata = band.GetNoDataValue()
        dataset = band.ReadAsArray()
        dataset = dataset.astype(float)
        dataset [dataset==nodata]= np.nan
        if band_tot > 1:
            data3d[:,:,i] = dataset
        else:
            data3d = dataset

    return data3d, xsize, ysize, geotransform, geoproj


def writeGeotiffSingleBand(filename, geotransform, geoprojection, data, nodata=np.nan, BandName= ""):
    (x, y) = data.shape
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    # dst_datatype = gdal.GDT_Byte #byte
    dst_datatype = gdal.GDT_Float32
    dst_ds = driver.Create(
        filename, y, x, 1, dst_datatype, options=[
            'COMPRESS=DEFLATE'])
    # sDATETIME= "2013:04:30 12:00:00"#The format is: "YYYY:MM:DD HH:MM:SS",
    # with hours like those on a 24-hour clock, and one space character
    # between the date and the time. The length of the string, including the
    # terminating NUL, is 20 bytes.
    #dst_ds.SetMetadata({'TIFFTAG_SOFTWARE': 'Hydra2D'})
    dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds.GetRasterBand(1).SetDescription (BandName)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoprojection)
    return 1


def writeGeotiff(filename, geotransform, geoprojection, data, nodata=np.nan, BandNames=None, globalDescr=""):

    dim = len(data.shape)
    if dim == 1:
        (x, y) = data.shape
        iNbands = 1
    else:
        (x, y, z) = data.shape
        iNbands = z

    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    # dst_datatype = gdal.GDT_Byte #byte
    dst_datatype = gdal.GDT_Float32
    # dst_ds = driver.Create(filename,y,x,1,dst_datatype,options = [
    # 'COMPRESS=DEFLATE', 'PREDICTOR=3' ]) #incompatibility of PREDICTOR con
    # Geoserver 2.2.4 (Bolivia)
    dst_ds = driver.Create(filename, y,x,iNbands, dst_datatype, options=['COMPRESS=DEFLATE'])
    dst_ds.SetDescription(globalDescr)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoprojection)
    # write data
    if iNbands == 1:
        dst_ds.GetRasterBand(1).WriteArray(data[:, :])
        dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
        if BandNames != None:
            dst_ds.GetRasterBand(1).SetDescription (BandNames)
    else:
        for i in range(0, iNbands):
            dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
            dst_ds.GetRasterBand(i + 1).WriteArray(data[:, :, i])
            if BandNames != None:
                dst_ds.GetRasterBand(i + 1).SetDescription (BandNames[i])
    return 1
