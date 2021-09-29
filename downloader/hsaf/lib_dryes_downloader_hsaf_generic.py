"""
Library Features:

Name:          lib_dryes_downloader_generic
Author(s):     Francesco Avanzi (francesco.avanzi@cimafoundation.org), Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210929'
Version:       '1.0.0'
"""
#################################################################################
# Library
import os
import re
import logging
import pandas as pd
from datetime import datetime
from copy import deepcopy
import xarray as xr
#################################################################################


# -------------------------------------------------------------------------------------
# Method to create a data array
def create_darray_3d(data, time, geo_x, geo_y, geo_1d=True, var_name=None,
                     coord_name_x='west_east', coord_name_y='south_north', coord_name_time='time',
                     dim_name_x='west_east', dim_name_y='south_north', dim_name_time='time',
                     dims_order=None):

    if dims_order is None:
        dims_order = [dim_name_y, dim_name_x, dim_name_time]

    if geo_1d:
        if geo_x.shape.__len__() == 2:
            geo_x = geo_x[0, :]
        if geo_y.shape.__len__() == 2:
            geo_y = geo_y[:, 0]

        data_da = xr.DataArray(data,
                               name=var_name,
                               dims=dims_order,
                               coords={coord_name_time: (dim_name_time, time),
                                       coord_name_x: (dim_name_x, geo_x),
                                       coord_name_y: (dim_name_y, geo_y)})
    else:
        logging.error(' ===> Longitude and Latitude must be 1d')
        raise IOError('Variable shape is not valid')

    return data_da
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to create a data array
def create_darray_2d(data, geo_x, geo_y, geo_1d=True, name='geo',
                     coord_name_x='west_east', coord_name_y='south_north',
                     dim_name_x='west_east', dim_name_y='south_north',
                     dims_order=None):

    if dims_order is None:
        dims_order = [dim_name_y, dim_name_x]

    if geo_1d:
        if geo_x.shape.__len__() == 2:
            geo_x = geo_x[0, :]
        if geo_y.shape.__len__() == 2:
            geo_y = geo_y[:, 0]

        data_da = xr.DataArray(data,
                               dims=dims_order,
                               coords={coord_name_x: (dim_name_x, geo_x),
                                       coord_name_y: (dim_name_y, geo_y)},
                               name=name)
    else:
        logging.error(' ===> Longitude and Latitude must be 1d')
        raise IOError('Variable shape is not valid')

    return data_da
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to add format(s) string (path or filename)
def fill_tags2string(string_raw, tags_format=None, tags_filling=None, tags_template='[TMPL_TAG_{:}]'):

    apply_tags = False
    if string_raw is not None:
        for tag in list(tags_format.keys()):
            if tag in string_raw:
                apply_tags = True
                break

    if apply_tags:

        string_filled = None
        tag_dictionary = {}
        for tag_id, (tag_key, tag_value) in enumerate(tags_format.items()):
            tag_key_tmp = '{' + tag_key + '}'
            if tag_value is not None:

                tag_id = tags_template.format(tag_id)
                tag_dictionary[tag_id] = {'key': None, 'value': None}

                if tag_key_tmp in string_raw:
                    tag_dictionary[tag_id] = {'key': tag_key, 'value': tag_value}
                    string_filled = string_raw.replace(tag_key_tmp, tag_id)
                    string_raw = string_filled
                else:
                    tag_dictionary[tag_id] = {'key': tag_key, 'value': None}

        dim_max = 1
        for tags_filling_values_tmp in tags_filling.values():
            if isinstance(tags_filling_values_tmp, list):
                dim_tmp = tags_filling_values_tmp.__len__()
                if dim_tmp > dim_max:
                    dim_max = dim_tmp

        string_filled_list = [string_filled] * dim_max

        string_filled_def = []
        for string_id, string_filled_step in enumerate(string_filled_list):

            for tag_dict_template, tag_dict_fields in tag_dictionary.items():
                tag_dict_key = tag_dict_fields['key']
                tag_dict_value = tag_dict_fields['value']

                if tag_dict_template in string_filled_step:
                    if tag_dict_value is not None:

                        if tag_dict_key in list(tags_filling.keys()):

                            value_filling_obj = tags_filling[tag_dict_key]

                            if isinstance(value_filling_obj, list):
                                value_filling = value_filling_obj[string_id]
                            else:
                                value_filling = value_filling_obj

                            string_filled_step = string_filled_step.replace(tag_dict_template, tag_dict_key)

                            if isinstance(value_filling, datetime):
                                tag_dict_value = value_filling.strftime(tag_dict_value)
                            elif isinstance(value_filling, (float, int)):
                                tag_dict_value = tag_dict_key.format(value_filling)
                            else:
                                tag_dict_value = value_filling

                            string_filled_step = string_filled_step.replace(tag_dict_key, tag_dict_value)

            string_filled_def.append(string_filled_step)

        if dim_max == 1:
            string_filled_out = string_filled_def[0].replace('//', '/')
        else:
            string_filled_out = []
            for string_filled_tmp in string_filled_def:
                string_filled_out.append(string_filled_tmp.replace('//', '/'))

        return string_filled_out
    else:
        return string_raw
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to create data outcome list
def set_data_outcome(time_stamp_list, file_id, variable_obj=None, group_obj=None, file_obj=None,
                       ancillary_obj=None, template_obj=None, flag_cleaning_outcome=False):

    folder_raw = file_obj['folder'][file_id]
    filename_raw = file_obj['filename'][file_id]

    group_list = group_obj[file_id]
    variable_list = variable_obj[file_id]
    domain = ancillary_obj['domain']

    filename_list = {}
    for time_stamp in time_stamp_list:

        var_list = []
        for variable, group in zip(variable_list, group_list):

            time_step = time_stamp.to_pydatetime()
            template_values = {"domain": domain,
                               "var_name": variable,
                               "group_name": group,
                               "outcome_sub_path_time": time_step,
                               "outcome_datetime": time_step}

            folder_step = fill_tags2string(folder_raw, template_obj, template_values)
            filename_step = fill_tags2string(filename_raw, template_obj, template_values)
            path_step = os.path.join(folder_step, filename_step)

            if flag_cleaning_outcome:
                if os.path.exists(path_step):
                    os.remove(path_step)

            var_list.append(path_step)

        filename_list[time_stamp] = var_list

    return filename_list
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to create data ancillary list
def set_data_ancillary(time_stamp_list, file_id, variable_obj=None, file_obj=None,
                       ancillary_obj=None, template_obj=None,
                       flag_cleaning_ancillary_global=False, flag_cleaning_ancillary_domain=False):

    folder_global_raw = file_obj['global']['folder'][file_id]
    filename_global_raw = file_obj['global']['filename'][file_id]
    folder_domain_raw = file_obj['domain']['folder'][file_id]
    filename_domain_raw = file_obj['domain']['filename'][file_id]

    variable_list = variable_obj[file_id]
    domain = ancillary_obj['domain']

    filename_global_list = {}
    filename_domain_list = {}
    for time_stamp in time_stamp_list:

        var_global_list = []
        var_domain_list = []
        for variable in variable_list:

            time_step = time_stamp.to_pydatetime()
            template_values = {"domain": domain,
                               "var_name": variable,
                               "ancillary_sub_path_time": time_step,
                               "ancillary_datetime": time_step}

            folder_global_step = fill_tags2string(folder_global_raw, template_obj, template_values)
            filename_global_step = fill_tags2string(filename_global_raw, template_obj, template_values)
            path_global_step = os.path.join(folder_global_step, filename_global_step)

            folder_domain_step = fill_tags2string(folder_domain_raw, template_obj, template_values)
            filename_domain_step = fill_tags2string(filename_domain_raw, template_obj, template_values)
            path_domain_step = os.path.join(folder_domain_step, filename_domain_step)

            if flag_cleaning_ancillary_global:
                if os.path.exists(path_global_step):
                    os.remove(path_global_step)
            if flag_cleaning_ancillary_domain:
                if os.path.exists(path_domain_step):
                    os.remove(path_domain_step)

            var_global_list.append(path_global_step)
            var_domain_list.append(path_domain_step)

        filename_global_list[time_stamp] = var_global_list
        filename_domain_list[time_stamp] = var_domain_list

    return filename_global_list, filename_domain_list
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to create data source list
def set_data_source(file_id, filename_url,
                    file_obj=None, variable_obj=None, root_obj=None, ancillary_obj=None, template_obj=None,
                    filename_suffix='.h5', flag_cleaning_source=False):

    if not isinstance(filename_url, list):
        filename_url = [filename_url]

    folder_raw = file_obj['folder'][file_id]
    filename_raw = file_obj['filename'][file_id]
    domain = ancillary_obj['domain']

    variable_list = variable_obj[file_id]
    fileroot_raw = root_obj[file_id]

    time_stamp_list = []
    filename_list_url = []
    filename_obj_source = {}
    fileroot_obj_source = {}
    for filename_url_step in filename_url:

        if filename_url_step.endswith(filename_suffix):

            match_time = re.search(r'\d{4}\d{2}\d{2}\w\d{2}\d{2}\d{2}', filename_url_step)
            time_str = match_time.group()
            time_stamp = pd.Timestamp(time_str)

            time_step = time_stamp.to_pydatetime()
            template_values = {"domain": domain,
                               "source_sub_path_time": time_step,
                               "source_datetime": time_step}

            folder_step = fill_tags2string(folder_raw, template_obj, template_values)
            filename_step = fill_tags2string(filename_raw, template_obj, template_values)

            path_step = os.path.join(folder_step, filename_step)

            if flag_cleaning_source:
                if os.path.exists(path_step):
                    os.remove(path_step)

            time_stamp_list.append(time_stamp)
            filename_list_url.append(filename_url_step)

            if time_step not in list(filename_obj_source.keys()):
                filename_obj_source[time_step] = [path_step]
            else:
                logging.error(' ===> Time is always set in source obj')
                raise NotImplementedError('Merge filename(s) is not implemented yet')

            var_fileroot_list = []
            for variable in variable_list:
                template_values = {"domain": domain,
                                   "file_name": path_step,
                                   "var_name": variable,
                                   "source_sub_path_time": time_step,
                                   "source_datetime": time_step}

                fileroot_step = fill_tags2string(fileroot_raw, template_obj, template_values)
                var_fileroot_list.append(fileroot_step)
            fileroot_obj_source[time_stamp] = var_fileroot_list

    return time_stamp_list, filename_list_url, filename_obj_source, fileroot_obj_source

# -------------------------------------------------------------------------------------

