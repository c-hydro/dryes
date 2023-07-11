"""
Class Features

Name:          lib_data_io_tiff
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210408'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import os

from copy import deepcopy

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import Affine

from tools.tool_processing_transfer_mcm.lib.lib_info_args import logger_name

from gdal import gdalconst
import gdal

# Logging
log_stream = logging.getLogger(logger_name)

# Debug
#######################################################################################

# -------------------------------------------------------------------------------------
# Default settings
proj_default_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
                   'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],' \
                   'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read data
def read_data_tiff(file_name, var_scale_factor=1, var_type='float32', var_name=None, var_time=None, var_no_data=-9999.0,
                   coord_name_time='time', coord_name_geo_x='Longitude', coord_name_geo_y='Latitude',
                   dim_name_time='time', dim_name_geo_x='west_east', dim_name_geo_y='south_north',
                   dims_order=None,
                   decimal_round_data=7, flag_round_data=False,
                   decimal_round_geo=7, flag_round_geo=True):

    if dims_order is None:
        dims_order = [dim_name_geo_y, dim_name_geo_x, dim_name_time]

    if os.path.exists(file_name):
        # Open file tiff
        file_handle = rasterio.open(file_name)
        # Read file info and values
        file_bounds = file_handle.bounds
        file_res = file_handle.res
        file_transform = file_handle.transform
        file_nodata = file_handle.nodata
        file_data = file_handle.read()
        file_values = file_data[0, :, :]

        if file_nodata is None:
            file_nodata = var_no_data

        if flag_round_data:
            file_values = file_values.round(decimal_round_data)

        if var_type == 'float64':
            file_values = np.float64(file_values / var_scale_factor)
        elif var_type == 'float32':
            file_values = np.float32(file_values / var_scale_factor)
        else:
            log_stream.error(' ===> File type is not correctly defined.')
            raise NotImplemented('Case not implemented yet')

        if file_handle.crs is None:
            file_proj = proj_default_wkt
            log_stream.warning(' ===> Projection of tiff ' + file_name + ' not defined. Use default settings.')
        else:
            file_proj = file_handle.crs.wkt

        file_geotrans = file_handle.transform

        file_dims = file_values.shape
        file_high = file_dims[0]
        file_wide = file_dims[1]

        center_right = file_bounds.right - (file_res[0] / 2)
        center_left = file_bounds.left + (file_res[0] / 2)
        center_top = file_bounds.top - (file_res[1] / 2)
        center_bottom = file_bounds.bottom + (file_res[1] / 2)

        if center_bottom > center_top:
            center_bottom_tmp = center_top
            center_top_tmp = center_bottom
            center_bottom = center_bottom_tmp
            center_top = center_top_tmp

            file_values = np.flipud(file_values)

        lon = np.arange(center_left, center_right + np.abs(file_res[0] / 2), np.abs(file_res[0]), float)
        lat = np.arange(center_bottom, center_top + np.abs(file_res[1] / 2), np.abs(file_res[1]), float)
        lons, lats = np.meshgrid(lon, lat)

        if flag_round_geo:
            min_lon_round = round(np.min(lons), decimal_round_geo)
            max_lon_round = round(np.max(lons), decimal_round_geo)
            min_lat_round = round(np.min(lats), decimal_round_geo)
            max_lat_round = round(np.max(lats), decimal_round_geo)

            center_right_round = round(center_right, decimal_round_geo)
            center_left_round = round(center_left, decimal_round_geo)
            center_bottom_round = round(center_bottom, decimal_round_geo)
            center_top_round = round(center_top, decimal_round_geo)
        else:
            log_stream.error(' ===> Switch off the rounding of geographical dataset is not expected')
            raise NotImplemented('Case not implemented yet')

        assert min_lon_round == center_left_round
        assert max_lon_round == center_right_round
        assert min_lat_round == center_bottom_round
        assert max_lat_round == center_top_round

        var_geo_x_2d = lons
        var_geo_y_2d = np.flipud(lats)

        var_data = np.zeros(shape=[var_geo_x_2d.shape[0], var_geo_y_2d.shape[1], 1])
        var_data[:, :, :] = np.nan

        var_data[:, :, 0] = file_values

        var_attrs = {'nrows': var_geo_y_2d.shape[0],
                     'ncols': var_geo_x_2d.shape[1],
                     'nodata_value': file_nodata,
                     'xllcorner': file_transform[2],
                     'yllcorner': file_transform[5],
                     'cellsize': round(abs(file_transform[0]),2),
                     'proj': file_proj,
                     'transform': file_geotrans}

    else:
        log_stream.warning(' ===> File ' + file_name + ' not available in loaded datasets!')
        var_data = None

    if var_data is not None:

        if var_time is not None:

            if isinstance(var_time, pd.Timestamp):
                var_time = pd.DatetimeIndex([var_time])
            elif isinstance(var_time, pd.DatetimeIndex):
                pass
            else:
                log_stream.error(' ===> Time format is not allowed. Expected Timestamp or Datetimeindex')
                raise NotImplemented('Case not implemented yet')

            var_da = xr.DataArray(var_data, name=var_name, dims=dims_order,
                                  coords={coord_name_time: ([dim_name_time], var_time),
                                          coord_name_geo_x: ([dim_name_geo_x], var_geo_x_2d[0, :]),
                                          coord_name_geo_y: ([dim_name_geo_y], var_geo_y_2d[:, 0])})

        else:

            var_notime = var_data[:,:,0].copy()
            dims_notime = dims_order[0:2]
            var_da = xr.DataArray(var_notime, name=var_name, dims=dims_notime,
                                  coords={coord_name_geo_x: ([dim_name_geo_x], var_geo_x_2d[0, :]),
                                          coord_name_geo_y: ([dim_name_geo_y], var_geo_y_2d[:, 0])})

        var_da.attrs = var_attrs

    else:
        log_stream.warning(' ===> All filenames in the selected period are not available')
        var_da = None

    return var_da
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to extract data grid
def extract_tiff_grid(file_name, tag_geo_values='data', tag_geo_x='geo_x', tag_geo_y='geo_y',
                      tag_nodata='nodata_value', value_no_data=-9999.0,
                      decimal_round_geo=7):

    if os.path.exists(file_name):
        # Open file tiff
        file_handle = rasterio.open(file_name)
        # Read file info and values
        file_bounds = file_handle.bounds
        file_res = file_handle.res
        height = file_handle.height
        width = file_handle.width
        file_data = file_handle.read()
        file_values = file_data[0, :, :]

        center_right = file_bounds.right - (file_res[0] / 2)
        center_left = file_bounds.left + (file_res[0] / 2)
        center_top = file_bounds.top - (file_res[1] / 2)
        center_bottom = file_bounds.bottom + (file_res[1] / 2)

        lon = np.arange(center_left, center_right + np.abs(file_res[0] / 2), np.abs(file_res[0]), float)
        lat = np.arange(center_bottom, center_top + np.abs(file_res[0] / 2), np.abs(file_res[1]), float)

        lons, lats = np.meshgrid(lon, lat)

        lats = np.flipud(lats)

        data_grid = {'nrows': height,
                     'ncols': width,
                     'xllcorner': round(file_bounds[0], decimal_round_geo),
                     'yllcorner': round(file_bounds[3], decimal_round_geo),
                     'cellsize': round(abs(file_res[0]), decimal_round_geo),
                     tag_geo_values: file_values,
                     tag_geo_x: lons[0, :],
                     tag_geo_y: lats[:, 0]}

        if tag_nodata not in list(data_grid.keys()):
            data_grid[tag_nodata] = value_no_data

    else:
        log_stream.warning(' ===> File ' + file_name + ' not available in loaded datasets!')
        data_grid = None

    return data_grid
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to set file tiff
def set_file_tiff(file_obj,
                  var_name_data='Rain', var_name_geo_x='longitude', var_name_geo_y='latitude',
                  var_no_data=-9999, var_limit_lower=0, var_limit_upper=None,
                  ):

    if isinstance(file_obj, xr.DataArray):
        file_tmp = deepcopy(file_obj)
    elif isinstance(file_obj, xr.Dataset):
        file_tmp = file_obj[var_name_data]
    else:
        log_stream.error(' ===> File obj format is not supported')
        raise NotImplemented('Case not implemented yet')

    file_data = file_tmp.values
    geo_x_obj = file_tmp[var_name_geo_x].values
    geo_y_obj = file_tmp[var_name_geo_y].values

    if geo_x_obj.ndim == 2:
        geo_x_arr = geo_x_obj[0, :]
    elif geo_x_obj.ndim == 1:
        geo_x_arr = deepcopy(geo_x_obj)
    else:
        log_stream.error(' ===> Geo X obj format is not supported')
        raise NotImplemented('Case not implemented yet')
    if geo_y_obj.ndim == 2:
        geo_y_arr = geo_y_obj[:, 0]
    elif geo_y_obj.ndim == 1:
        geo_y_arr = deepcopy(geo_y_obj)
    else:
        log_stream.error(' ===> Geo X obj format is not supported')
        raise NotImplemented('Case not implemented yet')

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

    file_data = np.flipud(file_data)

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
