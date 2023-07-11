"""
Class Features

Name:          driver_data_io_geo
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210408'
Version:       '1.0.0'
"""

######################################################################################
# Library
import logging
import os
from copy import deepcopy

from lib.lib_data_io_ascii import read_data_grid, \
    create_data_grid, extract_data_grid
from lib.lib_data_io_tiff import read_data_tiff, extract_tiff_grid
from lib.lib_data_io_nc import read_data_nc, parser_data_nc
from tools.tool_processing_transfer_mcm.lib.lib_info_args import logger_name, zip_extension
from lib.lib_utils_gzip import unzip_filename

# Logging
log_stream = logging.getLogger(logger_name)

# Debug
# import matplotlib.pylab as plt
######################################################################################


# -------------------------------------------------------------------------------------
# Class DriverStatic
class DriverStatic:

    # -------------------------------------------------------------------------------------
    # Initialize class
    def __init__(self, src_dict, dst_dict=None,
                 alg_ancillary=None, alg_template_tags=None,
                 flag_file_data='file_obj', flag_grid_data='grid_obj',
                 flag_static_source='source', flag_static_destination='destination',
                 flag_cleaning_static=True):

        self.flag_file_data = flag_file_data
        self.flag_grid_data = flag_grid_data

        self.flag_static_source = flag_static_source
        self.flag_static_destination = flag_static_destination

        self.alg_ancillary = alg_ancillary

        self.alg_template_tags = alg_template_tags
        self.file_var_tag = 'var_name'
        self.file_name_tag = 'file_name'
        self.folder_name_tag = 'folder_name'
        self.file_compression_tag = 'file_compression'
        self.file_type_tag = 'file_type'
        self.file_coords_tag = 'file_coords'

        self.file_obj_fields = [self.file_name_tag, self.folder_name_tag]
        self.grid_obj_fields = ["xll", "yll", "res", "nrows", "ncols"]

        self.obj_static_src = {}
        for src_key, src_fields in src_dict.items():

            if all(key in list(src_fields.keys()) for key in self.file_obj_fields):
                folder_name_src = src_fields[self.folder_name_tag]
                file_name_src = src_fields[self.file_name_tag]
                file_path_src = os.path.join(folder_name_src, file_name_src)

                file_compression = False
                if self.file_compression_tag in list(src_fields.keys()):
                    file_compression = src_fields[self.file_compression_tag]
                file_type = 'ascii'
                if self.file_type_tag in list(src_fields.keys()):
                    file_type = src_fields[self.file_type_tag]
                file_var = None
                if self.file_var_tag in list(src_fields.keys()):
                    file_var = src_fields[self.file_var_tag]
                file_coords = None
                if self.file_coords_tag in list(src_fields.keys()):
                    file_coords = src_fields[self.file_coords_tag]

                self.obj_static_src[src_key] = {}
                self.obj_static_src[src_key]['source'] = file_path_src
                self.obj_static_src[src_key]['obj'] = self.flag_file_data
                self.obj_static_src[src_key]['compression'] = file_compression
                self.obj_static_src[src_key]['type'] = file_type
                self.obj_static_src[src_key]['variable'] = file_var
                self.obj_static_src[src_key]['coords'] = file_coords

            elif all(key in list(src_fields.keys()) for key in self.grid_obj_fields):
                self.obj_static_src[src_key] = {}
                self.obj_static_src[src_key]['source'] = src_fields
                self.obj_static_src[src_key]['obj'] = self.flag_grid_data
                self.obj_static_src[src_key]['compression'] = None
                self.obj_static_src[src_key]['type'] = None
                self.obj_static_src[src_key]['variable'] = None
                self.obj_static_src[src_key]['coords'] = None
            else:
                log_stream.error(' ===> Static source key "' + src_key + '" is not in allowed format')
                raise NotImplementedError('Case not implemented yet')

        self.obj_static_dst = {}
        for dst_key, dst_fields in dst_dict.items():

            if all(key in list(dst_fields.keys()) for key in self.file_obj_fields):
                folder_name_src = dst_fields[self.folder_name_tag]
                file_name_src = dst_fields[self.file_name_tag]
                file_path_src = os.path.join(folder_name_src, file_name_src)

                file_compression = False
                if self.file_compression_tag in list(dst_fields.keys()):
                    file_compression = dst_fields[self.file_compression_tag]
                file_type = 'ascii'
                if self.file_type_tag in list(dst_fields.keys()):
                    file_type = dst_fields[self.file_type_tag]
                file_var = None
                if self.file_var_tag in list(dst_fields.keys()):
                    file_var = dst_fields[self.file_var_tag]
                file_coords = None
                if self.file_coords_tag in list(dst_fields.keys()):
                    file_coords = dst_fields[self.file_coords_tag]

                self.obj_static_dst[dst_key] = {}
                self.obj_static_dst[dst_key]['source'] = file_path_src
                self.obj_static_dst[dst_key]['obj'] = self.flag_file_data
                self.obj_static_dst[dst_key]['compression'] = file_compression
                self.obj_static_dst[dst_key]['type'] = file_type
                self.obj_static_dst[dst_key]['variable'] = file_var
                self.obj_static_dst[dst_key]['coords'] = file_coords

            elif all(key in list(dst_fields.keys()) for key in self.grid_obj_fields):
                self.obj_static_dst[dst_key] = {}
                self.obj_static_dst[dst_key]['source'] = dst_fields
                self.obj_static_dst[dst_key]['obj'] = self.flag_grid_data
                self.obj_static_dst[dst_key]['compression'] = None
                self.obj_static_dst[dst_key]['type'] = None
                self.obj_static_dst[dst_key]['variable'] = None
                self.obj_static_dst[dst_key]['coords'] = None
            else:
                log_stream.error(' ===> Static destination key "' + dst_key + '" is not in allowed format')
                raise NotImplementedError('Case not implemented yet')

        self.flag_cleaning_static = flag_cleaning_static

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to define unzipped filename(s)
    @staticmethod
    def define_file_name_unzip(file_path_zip, file_extension_zip=None):

        if file_extension_zip is None:
            file_extension_zip = zip_extension

        if file_path_zip.endswith(file_extension_zip):
            file_path_tmp, file_extension_tmp = os.path.splitext(file_path_zip)

            if file_extension_tmp.replace('.', '') == file_extension_zip.replace('.', ''):
                file_path_unzip = file_path_tmp
            else:
                log_stream.error(' ===> File zip extension was not expected in format "' + file_extension_tmp
                                 + '"; expected format was "' + file_extension_zip + '"')
                raise IOError('Check your settings file or change expected zip extension')
        else:
            log_stream.error(' ===> File zip ended with a unexpected zip extension "' + file_extension_tmp
                             + '"; expected format was "' + file_extension_zip + '"')
            raise IOError('Check your settings file or change expected zip extension')

        return file_path_unzip
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to extract variable field(s)
    def extract_var_fields(self, obj_collections):

        obj_source = obj_collections['source']
        obj_compression = obj_collections['compression']
        obj_format = obj_collections['obj']
        obj_type = obj_collections['type']
        obj_variable = obj_collections['variable']
        obj_coords = obj_collections['coords']

        return obj_source, obj_compression, obj_format, obj_type,  obj_variable, obj_coords
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to organize geographical data
    def organize_static(self):

        # Info start
        log_stream.info(' ---> Organize static datasets ... ')

        # Get static objects
        obj_static_src = self.obj_static_src
        obj_static_dst = self.obj_static_dst

        # Data collection object
        data_collections = {self.flag_static_source: {}, self.flag_static_destination: {}}

        # Read static data source
        for obj_key, obj_collections in obj_static_src.items():

            obj_data, obj_compression, \
                obj_format, obj_type, obj_variable, obj_coords = self.extract_var_fields(obj_collections)

            if obj_format == self.flag_file_data:

                obj_data_file_in = deepcopy(obj_data)
                if obj_compression:
                    obj_data_file_out = self.define_file_name_unzip(obj_data_file_in)
                    unzip_filename(obj_data_file_in, obj_data_file_out)
                else:
                    obj_data_file_out = deepcopy(obj_data_file_in)

                if obj_type == 'ascii':
                    obj_data_tmp = read_data_grid(obj_data_file_out, output_format='dictionary')
                    obj_data_grid = extract_data_grid(obj_data_tmp['values'],
                                                      obj_data_tmp['longitude'], obj_data_tmp['latitude'],
                                                      obj_data_tmp['transform'], obj_data_tmp['bbox'])
                elif obj_type == 'netcdf':

                    obj_dims = list(obj_coords.keys())[::-1]
                    obj_data_tmp = read_data_nc(obj_data_file_out, var_coords=obj_coords, dims_order=obj_dims,
                                                dim_name_geo_x=obj_dims[1], dim_name_geo_y=obj_dims[0],
                                                var_name=obj_variable)

                    obj_values, obj_geo_x, obj_geo_y, obj_geo_transform = parser_data_nc(obj_data_tmp)
                    obj_data_grid = extract_data_grid(obj_values, obj_geo_x, obj_geo_y, obj_geo_transform)

                elif obj_type == 'tiff':

                    obj_data_tmp = read_data_tiff(obj_data_file_out,
                                                  decimal_round_data=7,
                                                  decimal_round_geo=7,
                                                  flag_round_data=True)

                    obj_data_grid = extract_tiff_grid(obj_data_file_out, decimal_round_geo=2)

                else:
                    log_stream.error(' ===> Static source type "' + obj_type + '" is not allowed')
                    raise NotImplementedError('Case not implemented yet')

            elif obj_format == self.flag_grid_data:
                obj_data_fields = deepcopy(obj_data)
                obj_data_grid = create_data_grid(obj_data_fields)
            else:
                log_stream.error(' ===> Static source obj "' + str(obj_format) + '" is not allowed')
                raise NotImplementedError('Case not implemented yet')

            data_collections[self.flag_static_source][obj_key] = obj_data_grid

        # Read static data destination
        for obj_key, obj_collections in obj_static_dst.items():

            obj_data, obj_compression, \
                obj_format, obj_type, obj_variable, obj_coords = self.extract_var_fields(obj_collections)

            if obj_format == self.flag_file_data:

                obj_data_file_in = deepcopy(obj_data)
                if obj_compression:
                    obj_data_file_out = self.define_file_name_unzip(obj_data_file_in)
                    unzip_filename(obj_data_file_in, obj_data_file_out)
                else:
                    obj_data_file_out = deepcopy(obj_data_file_in)

                if obj_type == 'ascii':
                    obj_data_tmp = read_data_grid(obj_data_file_out, output_format='dictionary')
                    obj_data_grid = extract_data_grid(obj_data_tmp['values'],
                                                      obj_data_tmp['longitude'], obj_data_tmp['latitude'],
                                                      obj_data_tmp['transform'], obj_data_tmp['bbox'])
                elif obj_type == 'netcdf':

                    obj_dims = list(obj_coords.keys())[::-1]
                    obj_data_tmp = read_data_nc(obj_data_file_out, var_coords=obj_coords, dims_order=obj_dims,
                                                dim_name_geo_x=obj_dims[1], dim_name_geo_y=obj_dims[0],
                                                var_name=obj_variable)

                    obj_values, obj_geo_x, obj_geo_y, obj_geo_transform = parser_data_nc(obj_data_tmp)
                    obj_data_grid = extract_data_grid(obj_values, obj_geo_x, obj_geo_y, obj_geo_transform)

                elif obj_type == 'tiff':

                    obj_data_tmp = read_data_tiff(obj_data_file_out,
                                                  decimal_round_data=7,
                                                  decimal_round_geo=7,
                                                  flag_round_data=True)

                    obj_data_grid = extract_tiff_grid(obj_data_file_out, decimal_round_geo=2)

                else:
                    log_stream.error(' ===> Static destination type "' + obj_type + '" is not allowed')
                    raise NotImplementedError('Case not implemented yet')

            elif obj_format == self.flag_grid_data:
                obj_data_fields = deepcopy(obj_data)
                obj_data_grid = create_data_grid(obj_data_fields)
            else:
                log_stream.error(' ===> Static destination obj "' + str(obj_format) + '" is not allowed')
                raise NotImplementedError('Case not implemented yet')

            data_collections[self.flag_static_destination][obj_key] = obj_data_grid

        # Info end
        log_stream.info(' ---> Organize static datasets ... DONE')

        return data_collections
    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
