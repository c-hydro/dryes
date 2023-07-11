"""
Library Features:

Name:          lib_data_io_binary
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210408'
Version:       '1.0.0'
"""
#######################################################################################
# Library
import logging
import os
import struct
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from tools.tool_processing_transfer_mcm.lib.lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)

# Debug
#######################################################################################


# --------------------------------------------------------------------------------
# Method to read 2d variable in binary format (saved as 1d integer array)
def read_data_binary(file_name, var_geo_x, var_geo_y, var_geo_attrs=None,
                     var_format='i', var_scale_factor=10,
                     var_name=None, var_time=None, var_geo_1d=True, var_time_freq='H', var_time_steps_expected=1,
                     coord_name_geo_x='west_east', coord_name_geo_y='south_north', coord_name_time='time',
                     dim_name_geo_x='west_east', dim_name_geo_y='south_north', dim_name_time='time',
                     dims_order=None):

    if dims_order is None:
        dims_order = [dim_name_geo_y, dim_name_geo_x, dim_name_time]

    if os.path.exists(file_name):

        # Shape values 1d
        rows = var_geo_y.shape[0]
        cols = var_geo_x.shape[0]
        geo_n = rows * cols

        # Open and read binary file [OLD]
        file_handle = open(file_name, 'rb')
        data_format = var_format * geo_n
        data_stream = file_handle.read(-1)
        var_data_1d = struct.unpack(data_format, data_stream)
        file_handle.close()

        var_data_1d = np.asarray(var_data_1d, dtype=np.float32)
        var_data_1d = np.float32(var_data_1d / var_scale_factor)
        var_n = var_data_1d.shape[0]

        var_time_steps_cmp = int(var_n / geo_n)
        var_data_3d = np.reshape(var_data_1d, (rows, cols, var_time_steps_cmp), order='F')

        if var_geo_1d:
            var_geo_x_2d, var_geo_y_2d = np.meshgrid(var_geo_x, var_geo_y)
        else:
            var_geo_x_2d = var_geo_x
            var_geo_y_2d = var_geo_y

        geo_y_upper = var_geo_y_2d[0, 0]
        geo_y_lower = var_geo_y_2d[-1, 0]
        if geo_y_lower > geo_y_upper:
            var_geo_y_2d = np.flipud(var_geo_y_2d)

        var_dims = var_data_3d.shape
        var_high = var_dims[0]
        var_wide = var_dims[1]

        if var_time_steps_cmp == var_time_steps_expected:

            var_data = np.zeros(shape=[var_geo_x_2d.shape[0], var_geo_y_2d.shape[1], var_time_steps_cmp])
            var_data[:, :, :] = np.nan
            for step in np.arange(0, var_time_steps_cmp, 1):
                var_data_step = var_data_3d[:, :, step]
                var_data[:, :, step] = var_data_step

        elif (var_time_steps_cmp == 1) and (var_time_steps_cmp < var_time_steps_expected):

            log_stream.warning(' ===> File ' + file_name +
                               ' steps expected [' + str(var_time_steps_expected) +
                               '] and found [' + str(var_time_steps_cmp) + '] are different!')

            var_data = np.zeros(shape=[var_geo_x_2d.shape[0], var_geo_y_2d.shape[1], var_time_steps_expected])
            var_data[:, :, :] = np.nan
            for step in np.arange(0, var_time_steps_expected, 1):
                var_data_step = var_data_3d[:, :, 0]
                var_data[:, :, step] = var_data_step

            var_time_steps_cmp = deepcopy(var_time_steps_expected)

        else:
            log_stream.error(' ===> File ' + file_name + ' format are not expected!')
            raise NotImplemented('Case not implemented yet')

    else:
        log_stream.warning(' ===> File ' + file_name + ' not available in loaded datasets!')
        var_data = None

    if var_data is not None:

        if isinstance(var_time, pd.Timestamp):

            if var_time_steps_cmp == 1:
                var_time = pd.DatetimeIndex([var_time])
            elif var_time_steps_cmp > 1:
                var_time = pd.date_range(end=var_time, freq=var_time_freq, periods=var_time_steps_cmp)

        elif isinstance(var_time, pd.DatetimeIndex):
            pass
        else:
            log_stream.error(' ===> Time format is not allowed. Expected Timestamp or Datetimeindex')
            raise NotImplemented('Case not implemented yet')

        var_da = xr.DataArray(var_data, name=var_name, dims=dims_order,
                              coords={coord_name_time: ([dim_name_time], var_time),
                                      coord_name_geo_x: ([dim_name_geo_x], var_geo_x_2d[0, :]),
                                      coord_name_geo_y: ([dim_name_geo_y], var_geo_y_2d[:, 0])})
        if var_geo_attrs is not None:
            var_da.attrs = var_geo_attrs

    else:
        log_stream.warning(' ===> All filenames in the selected period are not available')
        var_da = None

    return var_da
# -------------------------------------------------------------------------------------
