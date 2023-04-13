__author__ = 'lauro'

from osgeo import gdal, gdalconst
import numpy as np

def rasterRegrid (sFileIn ,sFileMatch, sFileOut, method):
    ## resample sFileIn at the same resolution of sFileMatch

    if method == "nearest":
        interpMethod = gdalconst.GRA_NearestNeighbour
    elif method == "max":
        interpMethod = gdalconst.GRA_Max
    elif method == "average":
        interpMethod = gdalconst.GRA_Average
    else:
        interpMethod = gdalconst.GRA_NearestNeighbour
    #print (interpMethod)


    src = gdal.Open(sFileIn, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    
    # The source will be converted in order to match this:
    match_ds = gdal.Open(sFileMatch, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    print ("Regridded map size: ("+str(high)+","+str(wide)+")")

    # Output / destination
    dst = gdal.GetDriverByName('GTiff').Create(sFileOut, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)
    dst.GetRasterBand(1).SetNoDataValue(np.nan)
    ar = np.array([[np.nan]], dtype = np.float32)
    dst.GetRasterBand(1).WriteArray(ar)

    # gdal.ReprojectImage (src, dst, src_proj, src_proj, gdalconst.GRA_NearestNeighbour)
    gdal.ReprojectImage (src, dst, src_proj, match_proj, interpMethod)

    return match_geotrans,match_proj

"""
if __name__ == "__main__":

    sFileIn =  '/Users/lauro/Documents/PROJECTS/EU_ACP_UNISDR_Africa/Gambia/SFI.tif'
    sFileMatch = '/Users/lauro/Documents/PROJECTS/EU_ACP_UNISDR_Africa/Gambia/ESA_Gambia.tif'
    sFileOut = '/Users/lauro/Documents/PROJECTS/EU_ACP_UNISDR_Africa/Gambia/Gambia_SFI.tif'
    match_geotrans,match_proj=rasterRegrid (sFileIn ,sFileMatch, sFileOut, "average")
    print ('finish')

    from Geotiff import readFile, writeGeotiffSingleBand
    [xsize_orig, ysize_orig, geotransform, geoproj, data_mas] = readFile(sFileMatch)
    data_mas[data_mas<10]=0
    data_mas[data_mas>10]=0
    data_mas[data_mas==10]=1
    [xsize, ysize, geotransform, geoproj, data_high] = readFile(sFileOut)
    data_high = data_mas * data_high
    writeGeotiffSingleBand('/Users/lauro/Documents/PROJECTS/EU_ACP_UNISDR_Africa/Gambia/Gambia_SFI.tif',geotransform,geoproj,data_high)
"""