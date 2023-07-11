"""
Class Features

Name:          driver_data_io_dynamic
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210408'
Version:       '1.0.0'
"""

######################################################################################
# Library
import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from lib.lib_data_io_tiff import read_data_tiff, set_file_tiff, write_file_tiff
from tools.tool_processing_transfer_mcm.lib.lib_info_args import logger_name, \
    time_format_algorithm
from lib.lib_utils_interp import active_var_interp, apply_var_interp
from lib.lib_utils_io import read_obj, write_obj, create_dset
from tools.tool_processing_transfer_mcm.lib.lib_utils_system import fill_tags2string, make_folder

# Logging
log_stream = logging.getLogger(logger_name)

# Debug
######################################################################################

# -------------------------------------------------------------------------------------
# Default definition(s)
var_fields_accepted = [
    "var_compute", "var_name", "var_scale_factor",
    "folder_name", "file_name", "file_compression", "file_type", "file_frequency"]
time_format_reference = '%Y-%m-%d'
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Class DriverDynamic
class DriverDynamic:

    # -------------------------------------------------------------------------------------
    # Initialize class
    def __init__(self, time_reference, time_period,
                 time_run,
                 src_dict, anc_dict=None, dst_dict=None,
                 static_data_collection=None, interp_method='nearest',
                 alg_ancillary=None, alg_template_tags=None,
                 tag_terrain_data='Terrain', tag_grid_data='Grid',
                 tag_static_source='source', tag_static_destination='destination',
                 tag_dynamic_source='source', tag_dynamic_destination='destination',
                 tag_dynamic_anc_source='hourly', tag_dynamic_anc_destination='accumulated',
                 flag_cleaning_dynamic_ancillary=True,
                 flag_cleaning_dynamic_data=True,
                 flag_cleaning_dynamic_tmp=True,
                 flag_mask=True):

        self.time_str = time_reference.strftime(time_format_reference)
        self.time_period = time_period
        self.time_period_dest = time_run

        self.src_dict = src_dict
        self.anc_dict = anc_dict
        self.dst_dict = dst_dict

        self.tag_terrain_data = tag_terrain_data
        self.tag_grid_data = tag_grid_data

        self.tag_static_source = tag_static_source
        self.tag_static_destination = tag_static_destination
        self.tag_dynamic_source = tag_dynamic_source
        self.tag_dynamic_destination = tag_dynamic_destination

        self.alg_ancillary = alg_ancillary
        self.alg_template_tags = alg_template_tags

        self.static_data_src = static_data_collection[self.tag_static_source]
        self.static_data_dst = static_data_collection[self.tag_static_destination]

        self.rows_data_src = self.static_data_src['Terrain']['nrows']
        self.cols_data_src = self.static_data_src['Terrain']['ncols']

        self.var_compute_tag = 'var_compute'
        self.var_name_tag = 'var_name'
        self.var_scale_factor_tag = 'var_scale_factor'
        self.var_format_tag = 'var_format'
        self.file_name_tag = 'file_name'
        self.folder_name_tag = 'folder_name'
        self.file_compression_tag = 'file_compression'
        self.file_geo_reference_tag = 'file_geo_reference'
        self.file_type_tag = 'file_type'
        self.file_coords_tag = 'file_coords'
        self.file_frequency_tag = 'file_frequency'
        self.file_time_steps_expected_tag = 'file_time_steps_expected'
        self.file_time_steps_ref_tag = 'file_time_steps_ref'
        self.file_time_steps_flag_tag = 'file_time_steps_flag'

        self.alg_template_list = list(self.alg_template_tags.keys())
        self.var_name_obj = self.define_var_name(src_dict)
        self.file_path_obj_src = self.define_file_name_struct(self.src_dict, self.var_name_obj, self.time_period)
        self.file_path_obj_dst = self.define_file_name_struct(self.dst_dict, 'all', self.time_period)['all']
        self.file_path_obj_anc_source = self.define_file_name_struct(self.anc_dict[tag_dynamic_anc_source], 'all', self.time_period)['all']
        self.file_path_obj_anc_destination = self.define_file_name_struct(self.anc_dict[tag_dynamic_anc_destination], 'all', self.time_period_dest)['all']

        self.flag_cleaning_dynamic_ancillary = flag_cleaning_dynamic_ancillary
        self.flag_cleaning_dynamic_data = flag_cleaning_dynamic_data
        self.flag_cleaning_dynamic_tmp = flag_cleaning_dynamic_tmp
        self.flag_mask = flag_mask

        self.coord_name_geo_x = 'longitude'
        self.coord_name_geo_y = 'latitude'
        self.coord_name_time = 'time'
        self.dim_name_geo_x = 'longitude'
        self.dim_name_geo_y = 'latitude'
        self.dim_name_time = 'time'

        self.dims_order_2d = [self.dim_name_geo_y, self.dim_name_geo_x]
        self.dims_order_3d = [self.dim_name_geo_y, self.dim_name_geo_x, self.dim_name_time]
        self.coord_order_2d = [self.coord_name_geo_y, self.coord_name_geo_x]
        self.coord_order_3d = [self.coord_name_geo_y, self.coord_name_geo_x, self.coord_name_time]

        self.geo_da_dst = self.set_geo_reference()

        self.interp_method = interp_method

        self.thr_value = 3 # soglia sul numero di dati mancanti

        # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to set geographical reference
    def set_geo_reference(self):

        geo_ref_name = self.dst_dict[self.file_geo_reference_tag]
        geo_ref_collections = self.static_data_dst[geo_ref_name]

        geo_ref_data = geo_ref_collections['data']
        geo_ref_coord_x = geo_ref_collections['geo_x']
        geo_ref_coord_y = geo_ref_collections['geo_y']

        geo_ref_nrows = geo_ref_collections['nrows']
        geo_ref_ncols = geo_ref_collections['ncols']
        geo_ref_xll_corner = geo_ref_collections['xllcorner']
        geo_ref_yll_corner = geo_ref_collections['yllcorner']
        geo_ref_cellsize = geo_ref_collections['cellsize']
        geo_ref_nodata = geo_ref_collections['nodata_value']

        geo_ref_coord_x_2d, geo_ref_coord_y_2d = np.meshgrid(geo_ref_coord_x, geo_ref_coord_y)

        geo_y_upper = geo_ref_coord_y_2d[0, 0]
        geo_y_lower = geo_ref_coord_y_2d[-1, 0]
        if geo_y_lower > geo_y_upper:
            geo_ref_coord_y_2d = np.flipud(geo_ref_coord_y_2d)
            geo_ref_data = np.flipud(geo_ref_data)

        geo_da = xr.DataArray(
            geo_ref_data, name=geo_ref_name, dims=self.dims_order_2d,
            coords={self.coord_name_geo_x: (self.dim_name_geo_x, geo_ref_coord_x_2d[0, :]),
                    self.coord_name_geo_y: (self.dim_name_geo_y, geo_ref_coord_y_2d[:, 0])})

        geo_da.attrs = {'ncols': geo_ref_ncols, 'nrows': geo_ref_nrows,
                        'nodata_value': geo_ref_nodata,
                        'xllcorner': geo_ref_xll_corner, 'yllcorner': geo_ref_yll_corner,
                        'cellsize': geo_ref_cellsize}

        return geo_da
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to set geographical attributes
    @staticmethod
    def set_geo_attributes(dict_info, tag_data='data', tag_geo_x='geo_x', tag_geo_y='geo_y'):

        if tag_data in list(dict_info.keys()):
            data_values = dict_info[tag_data]
        else:
            log_stream.error(' ===> Tag "' + tag_data + '" is not available. Values are not found')
            raise IOError('Check your static datasets')
        if tag_geo_x in list(dict_info.keys()):
            data_geo_x = dict_info[tag_geo_x]
        else:
            log_stream.error(' ===> Tag "' + tag_geo_x + '" is not available. Values are not found')
            raise IOError('Check your static datasets')
        if tag_geo_y in list(dict_info.keys()):
            data_geo_y = dict_info[tag_geo_y]
        else:
            log_stream.error(' ===> Tag "' + tag_geo_y + '" is not available. Values are not found')
            raise IOError('Check your static datasets')

        data_attrs = deepcopy(dict_info)
        [data_attrs.pop(key) for key in [tag_data, tag_geo_x, tag_geo_y]]

        return data_values, data_geo_x, data_geo_y, data_attrs
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to define filename(s) struct
    def define_file_name_struct(self, var_dict, var_list, time_period):

        alg_template_tags = self.alg_template_tags

        if not isinstance(var_list, list):
            var_list = [var_list]

        file_path_dict = {}
        for var_name in var_list:

            if var_name in list(var_dict.keys()):
                folder_name_step = var_dict[var_name][self.folder_name_tag]
                file_name_step = var_dict[var_name][self.file_name_tag]
            else:
                folder_name_step = var_dict[self.folder_name_tag]
                file_name_step = var_dict[self.file_name_tag]

            file_path_step = os.path.join(folder_name_step, file_name_step)

            if isinstance(time_period, pd.DatetimeIndex):
                file_path_obj = []
                for time_step in time_period:
                    alg_template_filled = {}
                    for alg_template_step in self.alg_template_list:
                        alg_template_filled[alg_template_step] = time_step
                    file_path_obj.append(fill_tags2string(file_path_step, alg_template_tags, alg_template_filled))
            elif isinstance(time_period, pd.Timestamp):
                alg_template_filled = {}
                for alg_template_step in self.alg_template_list:
                    alg_template_filled[alg_template_step] = time_period
                file_path_obj = fill_tags2string(file_path_step, alg_template_tags, alg_template_filled)
            else:
                log_stream.error(' ===> Time obj must be Datetimeindex or Timestamp')
                raise NotImplemented('Case not implemented yet')

            file_path_dict[var_name] = file_path_obj

        return file_path_dict

    # -------------------------------------------------------------------------------------
    # Method to define variable names
    @staticmethod
    def define_var_name(data_dict, data_fields_excluded=None):

        if data_fields_excluded is None:
            data_fields_excluded = ['__comment__', '_comment_', 'comment','']

        var_list_tmp = list(data_dict.keys())
        var_list_def = [var_name for var_name in var_list_tmp if var_name not in data_fields_excluded]

        return var_list_def
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to extract variable field(s)
    def extract_var_fields(self, var_dict):

        var_compute = var_dict[self.var_compute_tag]
        var_name = var_dict[self.var_name_tag]
        var_scale_factor = var_dict[self.var_scale_factor_tag]
        file_compression = var_dict[self.file_compression_tag]
        file_geo_reference = var_dict[self.file_geo_reference_tag]
        file_type = var_dict[self.file_type_tag]
        file_coords = var_dict[self.file_coords_tag]
        file_freq = var_dict[self.file_frequency_tag]

        if self.var_format_tag in list(var_dict.keys()):
            var_format = var_dict[self.var_format_tag]
        else:
            if file_type == 'binary':
                var_format = 'i'
            else:
                var_format = None

        if self.file_time_steps_expected_tag in list(var_dict.keys()):
            file_time_steps_expected = var_dict[self.file_time_steps_expected_tag]
        else:
            file_time_steps_expected = 1
        if self.file_time_steps_ref_tag in list(var_dict.keys()):
            file_time_steps_ref = var_dict[self.file_time_steps_ref_tag]
        else:
            file_time_steps_ref = 1
        if self.file_time_steps_flag_tag in list(var_dict.keys()):
            file_time_steps_flag = var_dict[self.file_time_steps_flag_tag]
        else:
            file_time_steps_flag = self.dim_name_time

        return var_compute, var_name, var_scale_factor, var_format, \
               file_compression, file_geo_reference, file_type, file_coords, file_freq, \
               file_time_steps_expected, file_time_steps_ref, file_time_steps_flag
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to clean dynamic tmp
    def clean_dynamic_tmp(self):

        flag_cleaning_tmp = self.flag_cleaning_dynamic_tmp
        file_path_anc_source = self.file_path_obj_anc_source
        file_path_anc_destination = self.file_path_obj_anc_destination

        if isinstance(file_path_anc_destination, str):
            file_path_anc_destination = [file_path_anc_destination]
        else:
            file_path_anc_destination = file_path_anc_destination

        if flag_cleaning_tmp:
            for file_path_step in file_path_anc_source:
                if os.path.exists(file_path_step):
                    os.remove(file_path_step)
            for file_path_step in file_path_anc_destination:
                if os.path.exists(file_path_step):
                    os.remove(file_path_step)

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to compute dynamic data
    def compute_dynamic_data(self, var_name='Rain'):

        time_str = self.time_str
        time_period = self.time_period

        file_path_obj_anc_source = self.file_path_obj_anc_source
        file_path_obj_anc_destination = self.file_path_obj_anc_destination

        log_stream.info(' ---> Compute dynamic datasets [' + time_str + '] ... ')

        da_collections, thr_idx = None, 0
        for time_step, file_path_anc_source in zip(time_period, file_path_obj_anc_source):

            log_stream.info(' -----> Time "' + time_step.strftime(time_format_algorithm) + '" ... ')

            if os.path.exists(file_path_anc_source):

                dset_obj = read_obj(file_path_anc_source)
                da_obj = dset_obj[var_name]

                if da_collections is None:
                    da_collections = deepcopy(da_obj)
                else:
                    da_collections = xr.concat([da_collections, da_obj], dim='time')

                log_stream.info(' -----> Time "' + time_step.strftime(time_format_algorithm) + '" ... DONE')
            else:
                thr_idx += 1
                log_stream.warning(' ===> Hourly data missing')
                log_stream.info(' -----> Time "' + time_step.strftime(time_format_algorithm) + '" ... SKIPPED')

        # check the no data threshold
        if thr_idx < self.thr_value:

            da_accumulated = da_collections.sum(dim='time')



            folder_name_tmp, file_name_tmp = os.path.split(file_path_obj_anc_destination)
            if not os.path.exists(folder_name_tmp):
                make_folder(folder_name_tmp)

            write_obj(file_path_obj_anc_destination, da_accumulated)
        else:
            log_stream.warning(' ===> More than 3 hours lack in a day')
            log_stream.info(' -----> Time "' + time_step.strftime(time_format_algorithm) + '" ... SKIPPED')

        log_stream.info(' ---> Compute dynamic datasets [' + time_str + '] ... DONE')

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to dump dynamic data
    def dump_dynamic_data(self):

        time_str = self.time_str
        time_period = self.time_period

        var_name_obj = self.var_name_obj
        src_dict = self.src_dict
        dst_dict = self.dst_dict

        file_path_obj_anc = self.file_path_obj_anc_destination
        file_path_obj_dst = self.file_path_obj_dst

        if(isinstance(file_path_obj_anc, str)):
            file_path_obj_anc = [file_path_obj_anc]
        else:
            file_path_obj_anc = file_path_obj_anc

        flag_cleaning_dynamic = self.flag_cleaning_dynamic_data

        log_stream.info(' ---> Dump dynamic datasets [' + time_str + '] ... ')

        for time_step, file_path_anc, file_path_dst in zip(time_period, file_path_obj_anc, file_path_obj_dst):

            log_stream.info(' -----> Time "' + time_step.strftime(time_format_algorithm) + '" ... ')

            if flag_cleaning_dynamic:
                if os.path.exists(file_path_dst):
                    os.remove(file_path_dst)

            if os.path.exists(file_path_anc):

                dset_obj = read_obj(file_path_anc)

                folder_name_dst, file_name_dst = os.path.split(file_path_dst)
                if not os.path.exists(folder_name_dst):
                    make_folder(folder_name_dst)

                log_stream.info(' ------> Save filename "' + file_name_dst + '" ... ')

                if not (os.path.exists(file_path_dst)):

                     # Squeeze time dimensions (if equal == 1)  continuum expects 2d variables in forcing variables
                    if self.dim_name_time in list(dset_obj.dims):
                        time_array = dset_obj[self.dim_name_time].values
                        if time_array.shape[0] == 1:
                            dset_obj = dset_obj.squeeze(self.dim_name_time)
                            dset_obj = dset_obj.drop(self.dim_name_time)

                    log_stream.info(' --------> Write geotiff ... ')
                    file_data, file_data_width, file_data_height, \
                        file_data_transform, file_data_epsg_code = set_file_tiff(dset_obj, var_name_data='Rain')

                    write_file_tiff(file_path_dst, file_data, file_data_width, file_data_height,
                                    file_data_transform, file_data_epsg_code)
                    log_stream.info(' --------> Write geotiff ... DONE')

                    log_stream.info(' ------> Save filename "' + file_name_dst + '" ... DONE')

                else:
                    log_stream.info(' ------> Save filename "' + file_name_dst +
                                    '" ... SKIPPED. Filename previously saved')

                log_stream.info(' -----> Time "' + time_step.strftime(time_format_algorithm) + '" ... DONE')

            else:
                log_stream.info(' -----> Time "' + time_step.strftime(time_format_algorithm) +
                                '" ... SKIPPED. Datasets not available')

        log_stream.info(' ---> Dump dynamic datasets [' + time_str + '] ... DONE')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to organize dynamic data
    def organize_dynamic_data(self):

        time_str = self.time_str
        time_period = self.time_period

        geo_da_dst = self.geo_da_dst
        src_dict = self.src_dict

        var_name_obj = self.var_name_obj
        file_path_obj_src = self.file_path_obj_src
        file_path_obj_anc_source = self.file_path_obj_anc_source
        file_path_obj_anc_destination = self.file_path_obj_anc_destination

        flag_cleaning_ancillary = self.flag_cleaning_dynamic_ancillary
        flag_mask = self.flag_mask

        log_stream.info(' ---> Organize dynamic datasets [' + time_str + '] ... ')

        file_check_list = []
        for file_path_tmp in file_path_obj_anc_source:
            if os.path.exists(file_path_tmp):
                if flag_cleaning_ancillary:
                    os.remove(file_path_tmp)
                    file_check_list.append(False)
                else:
                    file_check_list.append(True)
            else:
                file_check_list.append(False)

        if isinstance(file_path_obj_anc_destination, str):
            file_path_obj_anc_destination_cleaner = [file_path_obj_anc_destination]
        else:
            file_path_obj_anc_destination_cleaner = file_path_obj_anc_destination

        for file_path_tmp in file_path_obj_anc_destination_cleaner:
            if os.path.exists(file_path_tmp):
                if flag_cleaning_ancillary:
                    os.remove(file_path_tmp)
                    file_check_list.append(False)
                else:
                    file_check_list.append(True)
            else:
                file_check_list.append(False)

        file_check = all(file_check_list)

        # Check elements availability
        if not file_check:

            dset_collection = {}
            for var_name in var_name_obj:

                log_stream.info(' ----> Variable "' + var_name + '" ... ')

                var_compute, var_tag, var_scale_factor, var_format, file_compression, \
                    file_geo_reference, file_type, file_coords, file_freq, \
                    file_time_steps_expected, \
                    file_time_steps_ref, file_time_steps_flag = self.extract_var_fields(src_dict[var_name])
                var_file_path_src = file_path_obj_src[var_name]

                if var_compute:

                    var_geo_data = None
                    for var_time, var_file_path_in in zip(time_period, var_file_path_src):

                        log_stream.info(' -----> Time "' + var_time.strftime(time_format_algorithm) + '" ... ')

                        if os.path.exists(var_file_path_in):

                            var_file_path_out = deepcopy(var_file_path_in)

                            if file_type == 'tiff':

                                var_da_src = read_data_tiff(
                                    var_file_path_out,
                                    var_scale_factor=var_scale_factor, var_name=var_tag, var_time=var_time,
                                    coord_name_geo_x=self.coord_name_geo_x, coord_name_geo_y=self.coord_name_geo_y,
                                    coord_name_time=self.coord_name_time,
                                    dim_name_geo_x=self.dim_name_geo_x, dim_name_geo_y=self.dim_name_geo_y,
                                    dim_name_time=self.dim_name_time,
                                    dims_order=self.dims_order_3d,
                                    decimal_round_data=7,
                                    decimal_round_geo=7,
                                    flag_round_data=True)

                            else:
                                log_stream.info(' -----> Time "' + var_time.strftime(time_format_algorithm) + '" ... FAILED')
                                log_stream.error(' ===> File type "' + file_type + '"is not allowed.')
                                raise NotImplementedError('Case not implemented yet')

                            # Delete (if needed the uncompressed file(s)
                            if var_file_path_in != var_file_path_out:
                                if os.path.exists(var_file_path_out):
                                    os.remove(var_file_path_out)

                            # Organize destination dataset
                            if var_da_src is not None:

                                # Active (if needed) interpolation method to the variable source data-array
                                active_interp = active_var_interp(var_da_src.attrs,
                                                                  geo_da_dst.attrs,
                                                                  fields_included=['ncols', 'nrows', 'cellsize'])

                                # Apply the interpolation method to the variable source data-array
                                if active_interp:
                                    var_da_dst = apply_var_interp(
                                        var_da_src, geo_da_dst,
                                        # var_name=var_name,
                                        dim_name_geo_x=self.dim_name_geo_x, dim_name_geo_y=self.dim_name_geo_y,
                                        coord_name_geo_x=self.coord_name_geo_x, coord_name_geo_y=self.coord_name_geo_y,
                                        interp_method=self.interp_method)
                                else:
                                    if var_tag != var_name:
                                        var_da_dst = deepcopy(var_da_src)
                                        var_da_dst.name = var_name
                                    else:
                                        var_da_dst = deepcopy(var_da_src)

                                # Mask the variable destination data-array
                                var_nodata = None
                                if 'nodata_value' in list(var_da_dst.attrs.keys()):
                                    var_nodata = var_da_dst.attrs['nodata_value']
                                geo_nodata = None
                                if 'nodata_value' in list(geo_da_dst.attrs.keys()):
                                    geo_nodata = geo_da_dst.attrs['nodata_value']

                                if geo_nodata is not None and (var_nodata is not None):
                                    var_da_masked = var_da_dst.where(
                                        (geo_da_dst.values[:, :, np.newaxis] != geo_nodata))
                                    var_da_masked = xr.where(var_da_masked.isnull(), geo_nodata, var_da_masked.values)

                                        # ((geo_da_dst.values[:, :, np.newaxis] != geo_nodata) != geo_nodata) &
                                        # (var_da_dst != var_nodata))
                                else:
                                    var_da_masked = deepcopy(var_da_dst)

                                # if geo_nodata is not None and (var_nodata is not None) and flag_mask:
                                #     var_da_masked = var_da_masked.where(((geo_da_dst.isnull)))

                                # '''
                                # plt.figure(1)
                                # plt.imshow(var_da_dst.values[:, :, 0])
                                # plt.colorbar()
                                # plt.figure(2)
                                # # plt.imshow(var_da_src.values[:, :, 0])
                                # # plt.colorbar()
                                # # plt.figure(3)
                                # plt.imshow(var_da_masked.values[:, :, 0])
                                # plt.colorbar()
                                # plt.show()
                                # plt.figure(4)
                                # plt.imshow(geo_da_dst.values)
                                # plt.colorbar()
                                # plt.show()
                                # '''

                                # Organize data in a common datasets
                                if self.dim_name_time in list(var_da_masked.dims):
                                    var_time_dset = pd.DatetimeIndex(var_da_masked[self.dim_name_time].values)
                                else:
                                    var_time_dset = deepcopy(var_time)

                                var_dset_masked = create_dset(var_data_time=var_time_dset,
                                                              var_data_name=var_name, var_data_values=var_da_masked,
                                                              var_data_attrs=None,
                                                              var_geo_1d=False,
                                                              file_attributes=geo_da_dst.attrs,
                                                              var_geo_name='terrain',
                                                              var_geo_values=geo_da_dst.values,
                                                              var_geo_x=geo_da_dst[self.coord_name_geo_x].values,
                                                              var_geo_y=geo_da_dst[self.coord_name_geo_y].values,
                                                              var_geo_attrs=None)

                                # Organize data in merged datasets
                                if var_time not in list(dset_collection.keys()):
                                    dset_collection[var_time] = var_dset_masked
                                else:
                                    var_dset_tmp = deepcopy(dset_collection[var_time])
                                    var_dset_tmp = var_dset_tmp.merge(var_dset_masked, join='right')
                                    dset_collection[var_time] = var_dset_tmp

                                log_stream.info(' -----> Time "' + var_time.strftime(time_format_algorithm) +
                                                '" ... DONE')

                            else:
                                log_stream.info(' -----> Time "' + var_time.strftime(time_format_algorithm) +
                                                '" ... SKIPPED. Datasets is not defined')
                        else:
                            log_stream.info(' -----> Time "' + var_time.strftime(time_format_algorithm) +
                                            '" ... SKIPPED. File source "' + var_file_path_in + '" is not available')

                    log_stream.info(' ----> Variable "' + var_name + '" ... DONE')

                else:
                    log_stream.info(' ----> Variable "' + var_name + '" ... SKIPPED. Compute flag not activated.')

            # Save ancillary datasets
            for file_path_anc_source, (dset_time, dset_anc) in zip(file_path_obj_anc_source, dset_collection.items()):

                folder_name_anc_source, file_name_anc_source = os.path.split(file_path_anc_source)
                if not os.path.exists(folder_name_anc_source):
                    make_folder(folder_name_anc_source)

                write_obj(file_path_anc_source, dset_anc)

            log_stream.info(' ---> Organize dynamic datasets [' + time_str + '] ... DONE')
        else:
            log_stream.info(' ---> Organize dynamic datasets [' +
                            time_str + '] ... SKIPPED. All datasets are previously computed')

    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
