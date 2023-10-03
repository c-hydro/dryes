#######################################################################################
# Library
import logging
import rasterio
import numpy as np
from rasterio.transform import Affine
from osgeo import gdal, gdalconst

#######################################################################################

# -------------------------------------------------------------------------------------
# Method to set file tiff
def set_file_tiff(file_darray,
                  var_name_data='surface_soil_moisture', var_name_geo_x='longitude', var_name_geo_y='latitude',
                  var_no_data=-9999, var_limit_lower=0, var_limit_upper=100,
                  ):

    file_data = file_darray.values
    geo_x_arr = file_darray[var_name_geo_x].values
    geo_y_arr = file_darray[var_name_geo_y].values

    geo_x_grid, geo_y_grid = np.meshgrid(geo_x_arr, geo_y_arr)

    file_data_height, file_data_width = file_data.shape
    file_geo_x_west = np.min(np.min(geo_x_grid))
    file_geo_x_east = np.max(np.max(geo_x_grid))
    file_geo_y_south = np.min(np.min(geo_y_grid))
    file_geo_y_north = np.max(np.max(geo_y_grid))

    if var_limit_lower is not None:
        file_data[file_data < var_limit_lower] = var_no_data
    if var_limit_upper is not None:
        file_data[file_data > var_limit_upper] = var_no_data
    # file_data[np.isnan(file_data)] = -9999

    #file_data = np.flipud(file_data)

    file_data_transform = rasterio.transform.from_bounds(
        file_geo_x_west, file_geo_y_south, file_geo_x_east, file_geo_y_north,
        file_data_width, file_data_height)

    file_data_epsg_code = 'EPSG:4326'

    return file_data, file_data_width, file_data_height, file_data_transform, file_data_epsg_code
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to write file tiff
def write_file_tiff(file_name, file_data, file_wide, file_high, file_geotrans, file_proj,
                    file_metadata=None, file_format=gdalconst.GDT_Float32):

    if not isinstance(file_data, list):
        file_data = [file_data]

    if file_metadata is None:
        file_metadata = {'description_field': 'data'}
    if not isinstance(file_metadata, list):
        file_metadata = [file_metadata] * file_data.__len__()

    if isinstance(file_geotrans, Affine):
        file_geotrans = file_geotrans.to_gdal()

    file_crs = rasterio.crs.CRS.from_string(file_proj)
    file_wkt = file_crs.to_wkt()

    file_n = file_data.__len__()
    dset_handle = gdal.GetDriverByName('GTiff').Create(file_name, file_wide, file_high, file_n, file_format,
                                                       options=['COMPRESS=DEFLATE'])
    dset_handle.SetGeoTransform(file_geotrans)
    dset_handle.SetProjection(file_wkt)

    for file_id, (file_data_step, file_metadata_step) in enumerate(zip(file_data, file_metadata)):
        dset_handle.GetRasterBand(file_id + 1).WriteArray(file_data_step)
        dset_handle.GetRasterBand(file_id + 1).SetMetadata(file_metadata_step)
    del dset_handle
# -------------------------------------------------------------------------------------
