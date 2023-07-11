"""
Class Features

Name:          lib_data_io_ascii
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210408'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from tools.tool_processing_transfer_mcm.lib.lib_info_args import logger_name
from tools.tool_processing_transfer_mcm.lib.lib_info_args import proj_epsg as proj_epsg_default
from lib.lib_utils_io import create_darray_2d

# Logging
log_stream = logging.getLogger(logger_name)

logging.getLogger("rasterio").setLevel(logging.WARNING)
# Debug
#######################################################################################

# -------------------------------------------------------------------------------------
# Default method variable(s)
info_default_fields = ['nrows', 'ncols', 'xll', 'yll', 'res']
map_default_fields = ['nrows', 'ncols', 'xllcorner', 'yllcorner', 'cellsize']
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to extract data grid information
def extract_data_grid(geo_data, geo_x_values, geo_y_values, geo_transform, geo_bbox=None,
                      tag_geo_values='data',
                      tag_geo_x='geo_x', tag_geo_y='geo_y',
                      tag_nodata='nodata_value', value_no_data=-9999.0):

    if geo_bbox is not None:
        geo_bbox_xll = geo_bbox[0]
        geo_bbox_yll = geo_bbox[1]
        data_grid = {'nrows': geo_y_values.shape[0], 'ncols': geo_x_values.shape[0], 'xllcorner': geo_bbox_xll,
                     'yllcorner': geo_bbox_yll, 'cellsize': abs(geo_transform[0]), tag_geo_values: geo_data,
                     tag_geo_x: geo_x_values, tag_geo_y: geo_y_values}
    elif geo_bbox is None:
        data_grid = {'nrows': geo_y_values.shape[0], 'ncols': geo_x_values.shape[0], 'xllcorner': geo_transform[0],
                     'yllcorner': geo_transform[5], 'cellsize': abs(geo_transform[1]), tag_geo_values: geo_data,
                     tag_geo_x: geo_x_values, tag_geo_y: geo_y_values}

    if tag_nodata not in list(data_grid.keys()):
        data_grid[tag_nodata] = value_no_data

    return data_grid
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to create data grid information
def create_data_grid(info_grid, info_expected_fields=None, map_expected_fields=None,
                     tag_geo_values='data',
                     tag_geo_x='geo_x', tag_geo_y='geo_y',
                     tag_nodata='nodata_value', value_no_data=-9999.0, value_default_data=1):

    if info_expected_fields is None:
        info_expected_fields = info_default_fields
    if map_expected_fields is None:
        map_expected_fields = map_default_fields

    data_grid = {}
    if info_grid is not None:
        if isinstance(info_grid, dict):
            if set(info_expected_fields).issubset(info_grid):
                if any(field is not None for field in info_grid.values()):
                    for info_field, map_field in zip(info_expected_fields, map_expected_fields):
                        data_grid[map_field] = info_grid[info_field]

                    xll = data_grid['xllcorner']
                    yll = data_grid['yllcorner']
                    nrows = data_grid['nrows']
                    ncols = data_grid['ncols']
                    res = data_grid['cellsize']

                    geo_x_start = xll + res / 2
                    geo_x_end = xll + res / 2 + res * (ncols - 1)
                    geo_x_values = np.linspace(geo_x_start, geo_x_end, ncols)

                    geo_y_start = yll + res / 2
                    geo_y_end = yll + res / 2 + res * (nrows - 1)
                    geo_y_values = np.linspace(geo_y_start, geo_y_end, nrows)

                    # geo_x_values = np.arange(xll + res / 2, xll + res / 2 + res * ncols, res)
                    # geo_y_values = np.arange(yll + res / 2, yll + res / 2 + res * nrows, res)

                    geo_x_values_2d, geo_y_values_2d = np.meshgrid(geo_x_values, geo_y_values)

                    geo_y_right = geo_x_values_2d[0, 0]
                    geo_y_left = geo_x_values_2d[0, -1]
                    geo_y_upper = geo_y_values_2d[0, 0]
                    geo_y_lower = geo_y_values_2d[-1, 0]
                    if geo_y_lower > geo_y_upper:
                        geo_y_values_2d = np.flipud(geo_y_values_2d)

                    geo_data_values = np.zeros([geo_y_values.shape[0], geo_x_values.shape[0]])
                    geo_data_values[:, :] = value_default_data

                    data_grid[tag_geo_values] = geo_data_values
                    data_grid[tag_geo_x] = geo_x_values_2d[0, :]
                    data_grid[tag_geo_y] = geo_y_values_2d[:, 0]

                    if tag_nodata not in list(data_grid.keys()):
                        data_grid[tag_nodata] = value_no_data
                else:
                    log_stream.warning(' ===> Grid information are not all defined. Datasets will be set to None')
                    data_grid = None
            else:
                log_stream.warning(' ===> Grid information are not enough. Datasets will be set to None')
                data_grid = None
        else:
            log_stream.warning(' ===> Grid information are not in dictionary format. Datasets will be set to None')
            data_grid = None
    else:
        log_stream.warning(' ===> Grid information are null. Datasets will be set to None')
        data_grid = None

    return data_grid
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read an ascii grid file
def read_data_grid(file_name, output_format='data_array', output_dtype='float32'):

    try:
        dset = rasterio.open(file_name)
        bounds = dset.bounds
        res = dset.res
        transform = dset.transform
        data = dset.read()

        if dset.crs is None:
            crs = CRS.from_string(proj_epsg_default)
        else:
            crs = dset.crs

        values = data[0, :, :]

        decimal_round = 7

        center_right = bounds.right - (res[0] / 2)
        center_left = bounds.left + (res[0] / 2)
        center_top = bounds.top - (res[1] / 2)
        center_bottom = bounds.bottom + (res[1] / 2)

        lon = np.arange(center_left, center_right + np.abs(res[0] / 2), np.abs(res[0]), float)
        lat = np.arange(center_bottom, center_top + np.abs(res[0] / 2), np.abs(res[1]), float)
        lons, lats = np.meshgrid(lon, lat)

        min_lon_round = round(np.min(lons), decimal_round)
        max_lon_round = round(np.max(lons), decimal_round)
        min_lat_round = round(np.min(lats), decimal_round)
        max_lat_round = round(np.max(lats), decimal_round)

        center_right_round = round(center_right, decimal_round)
        center_left_round = round(center_left, decimal_round)
        center_bottom_round = round(center_bottom, decimal_round)
        center_top_round = round(center_top, decimal_round)

        assert min_lon_round == center_left_round
        assert max_lon_round == center_right_round
        assert min_lat_round == center_bottom_round
        assert max_lat_round == center_top_round

        lats = np.flipud(lats)

        if output_format == 'data_array':

            data_obj = create_darray_2d(values, lons[0, :], lats[:, 0],
                                        coord_name_x='west_east', coord_name_y='south_north',
                                        dim_name_x='west_east', dim_name_y='south_north')

        elif output_format == 'dictionary':

            data_obj = {'values': values, 'longitude': lons[0, :], 'latitude': lats[:, 0],
                        'transform': transform, 'crs': crs,
                        'bbox': [bounds.left, bounds.bottom, bounds.right, bounds.top],
                        'bb_left': bounds.left, 'bb_right': bounds.right,
                        'bb_top': bounds.top, 'bb_bottom': bounds.bottom,
                        'res_lon': res[0], 'res_lat': res[1]}
        else:
            log_stream.error(' ===> File static "' + file_name + '" output format not allowed')
            raise NotImplementedError('Case not implemented yet')

    except IOError as io_error:

        data_obj = None
        log_stream.warning(' ===> File static in ascii grid was not correctly open with error "' + str(io_error) + '"')
        log_stream.warning(' ===> Filename "' + os.path.split(file_name)[1] + '"')

    return data_obj
# -------------------------------------------------------------------------------------
