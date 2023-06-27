#!/usr/bin/python3

"""
DRYES PROCESSING TOOLS - File Transfer

__date__ = '20230621'
__version__ = '1.0.0'
__author__ =
        'Michel Isabellon' (michel.isabellon@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org',

__library__ = 'dryes'

General command line:
python3 dryes_tool_transfer_datasets.py -settings_file configuration.json -time "YYYY-mm-dd HH:MM"
"""


# -------------------------------------------------------------------------------------
# Libraries
import logging
import os
import re
import subprocess
import time
import glob
from copy import deepcopy
from datetime import datetime

from lib.lib_utils_time import set_time
from lib.lib_info_args import logger_name, time_format_algorithm
from lib.lib_info_args import get_args
from lib.lib_utils_logging import set_logging_file
from lib.lib_data_io_json import read_file_settings

# Logging
log_stream = logging.getLogger(logger_name)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Script settings
tag_folder_name = 'folder_name'
tag_file_name = 'file_name'
tag_method = 'method'
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
project_name = 'DRYES'
alg_name = 'Datasets Transfer'
alg_type = 'Processing Tool'
alg_version = '1.0.0'
alg_release = '2023-06-01'
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to join file registry
def main():

    # -------------------------------------------------------------------------------------
    # Get and read algorithm settings
    file_configuration, time_run_args = get_args()

    settings_data = read_file_settings(file_configuration)

    # Configure log information
    folder_name_log, file_name_log = settings_data['log']['folder_name'], settings_data['log']['file_name']
    file_name_log = file_name_log.replace("{date}", datetime.today().strftime("%Y%m%d_%H%M"))
    set_logging_file(
        logger_name=logger_name,
        logger_file=os.path.join(folder_name_log, file_name_log))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    log_stream.info(' ============================================================================ ')
    log_stream.info('[' + project_name + ' ' + alg_type + ' - ' + alg_name + ' (Version ' + alg_version +
                    ' - Release ' + alg_release + ')]')
    log_stream.info(' ==> START ... ')
    log_stream.info(' ')

    # Time algorithm information
    alg_time_start = time.time()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm start
    log_stream.info(' ---> Transfer datasets from source to destination location(s) ... ')

    # Configure ancillary information
    ancillary_raw = settings_data['ancillary']

    # Configure template information
    template_time_raw = settings_data['template_time']
    template_string_raw = settings_data['template_string']

    # Configure time information
    time_run_file = settings_data['time']['time_run']
    time_start = settings_data['time']['time_start']
    time_end = settings_data['time']['time_end']
    time_period = settings_data['time']['time_period']
    time_frequency = settings_data['time']['time_frequency']
    time_rounding = settings_data['time']['time_rounding']
    time_change = settings_data['time']['time_mcm_change']

    time_change = datetime.strptime(time_change, '%Y-%m-%d %H:%M')
    time_run = datetime.strptime(time_run_args, '%Y-%m-%d %H:%M')

    # Configure datasets information
    if time_run > time_change:
        dataset_type = 'datasets_mcm_latest'
    else:
        dataset_type = 'datasets_mcm_first'

    data_src_settings = settings_data['source'][dataset_type]

    data_dst_settings = settings_data['destination']

    # Configure method(s)
    data_src_methods = settings_data['method']

    time_run, time_range, time_chunks = set_time(
        time_run_args=time_run_args,
        time_run_file=time_run_file,
        time_run_file_start=time_start,
        time_run_file_end=time_end,
        time_format=time_format_algorithm,
        time_period=time_period,
        time_frequency=time_frequency,
        time_rounding=time_rounding)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Iterate over time period
    for time_step in time_range:

        # -------------------------------------------------------------------------------------
        # Info time start
        log_stream.info(' ----> Time "' + time_step.strftime(format=time_format_algorithm) + '" ... ')

        # Iterate over datasets
        for (dset_key, dset_fields_src), dset_fields_dst in zip(data_src_settings.items(), data_dst_settings.values()):
            print(dset_key)
            print(dset_fields_src)
            print(dset_fields_dst)

            # Iterate over name list
            for tag_value in ancillary_raw['tag_name_list']:

                # Fill template for time variable(s)
                template_time_filled = {}
                for time_key, time_format in template_time_raw.items():
                    template_time_filled[time_key] = time_step.strftime(time_format)
                # Fill template for other variable(s)
                template_var_filled = {'tag_name': tag_value}

                # define generic template
                template_generic_filled = {**template_time_filled, **template_var_filled}

                # Filled the structure with
                dset_key_step = fill_object(data_structure_raw=dset_key,
                                            data_template=template_generic_filled)

                dset_fields_src_step = fill_object(data_structure_raw=dset_fields_src,
                                                   data_template=template_generic_filled)

                dset_fields_dst_step = fill_object(data_structure_raw=dset_fields_dst,
                                                   data_template=template_generic_filled)


                # Info dataset start
                log_stream.info(' -----> Dataset "' + dset_key_step + '" ... ')

                template_time_filled = {}
                for time_key, time_format in template_time_raw.items():
                    template_time_filled[time_key] = time_step.strftime(time_format)

                # Transfer method
                file_method_src = dset_fields_src_step[tag_method]

                if file_method_src in list(data_src_methods.keys()):
                    method_mode = data_src_methods['mode']
                    method_info = data_src_methods[file_method_src]['settings']
                    method_command_ancillary = data_src_methods[file_method_src]['command_ancillary']
                    method_command_exec = data_src_methods[file_method_src]['command_exec']
                    method_command_line = data_src_methods[file_method_src]['command_line']
                else:
                    log_stream.error(' ===> Method "' + file_method_src + ' is not defined in the settings file.')
                    raise IOError('Check your settings file and insert the method and its fields')

                # File path source
                folder_name_src_tmp = dset_fields_src_step[tag_folder_name]
                file_name_src_tmp = dset_fields_src_step[tag_file_name]
                file_path_src_tmp = os.path.join(folder_name_src_tmp, file_name_src_tmp)
                file_path_src_def = file_path_src_tmp.format(**template_time_filled)

                if '*' in file_path_src_def:
                    file_list_src_def = glob.glob(file_path_src_def)
                elif '*' not in file_path_src_def:
                    if isinstance(file_path_src_def, str):
                        file_list_src_def = [file_path_src_def]
                    elif isinstance(file_path_src_def, list):
                        file_list_src_def = deepcopy(file_path_src_def)
                    else:
                        log_stream.error(' ===> File format source is not in supported format')
                        raise NotImplementedError('Case not implemented yet')
                else:
                    file_list_src_def = deepcopy(file_path_src_def)

                # Check the list source file(s)
                if file_list_src_def:

                    # File path destination
                    folder_name_dst_tmp = dset_fields_dst_step[tag_folder_name]
                    file_name_dst_tmp = dset_fields_dst_step[tag_file_name]

                    if ('*' in file_name_src_tmp) and ('*' not in file_name_dst_tmp):
                        log_stream.warning(
                            ' ===> Symbol "*" defined in the source file(s), but not in the destination file(s).')
                        log_stream.warning(' ===> Destination file(s) are defined using the name of source file(s)')
                        file_name_dst_tmp = None

                    if (file_name_dst_tmp is not None) and ('*' not in file_name_dst_tmp):

                        file_path_dst_tmp = os.path.join(folder_name_dst_tmp, file_name_dst_tmp)
                        file_path_dst_def = file_path_dst_tmp.format(**template_time_filled)

                    elif (file_name_dst_tmp is None) and ('*' in file_path_src_def):

                        file_path_dst_def = []
                        for file_path_src_tmp in file_list_src_def:
                            folder_name_src_tmp, file_name_src_tmp = os.path.split(file_path_src_tmp)
                            file_path_dst_tmp = os.path.join(folder_name_dst_tmp, file_name_src_tmp)
                            file_path_dst_filled = file_path_dst_tmp.format(**template_time_filled)
                            file_path_dst_def.append(file_path_dst_filled)

                    elif (file_name_dst_tmp is not None) and ('*' in file_name_dst_tmp):
                        file_path_dst_def = []
                        for file_path_src_tmp in file_list_src_def:
                            folder_name_src_tmp, file_name_src_tmp = os.path.split(file_path_src_tmp)
                            file_path_dst_tmp = os.path.join(folder_name_dst_tmp, file_name_src_tmp)
                            file_path_dst_filled = file_path_dst_tmp.format(**template_time_filled)
                            file_path_dst_def.append(file_path_dst_filled)

                    else:
                        log_stream.error(' ===> File destination name is not defined')
                        raise NotImplementedError('Case not implemented yet')

                    if isinstance(file_path_dst_def, str):
                        file_list_dst_def = [file_path_dst_def]
                    elif isinstance(file_path_dst_def, list):
                        file_list_dst_def = deepcopy(file_path_dst_def)
                    else:
                        log_stream.error(' ===> File format source is not in supported format')
                        raise NotImplementedError('Case not implemented yet')

                    # Cycles over source and destination file(s)
                    for file_path_src_step, file_path_dst_step in zip(file_list_src_def, file_list_dst_def):
                        print(file_path_src_step)
                        print(file_path_dst_step)

                        # Define folder and file name(s)
                        folder_name_src_step, file_name_src_step = os.path.split(file_path_src_step)
                        folder_name_dst_step, file_name_dst_step = os.path.split(file_path_dst_step)

                        # Method settings
                        file_info = {
                            'folder_name_src': folder_name_src_step, 'file_name_src': file_name_src_step,
                            'folder_name_dst': folder_name_dst_step, 'file_name_dst': file_name_dst_step}

                        template_command_line = {**method_info, **file_info}

                        method_cmd_create_folder, method_cmd_uncompress_file = None, None
                        method_cmd_find_file, method_cmd_remove_file = None, None
                        if (method_mode == 'local2local') or (method_mode == 'remote2local'):
                            make_folder(folder_name_dst_step)
                        elif method_mode == 'local2remote':

                            if 'create_folder' in list(method_command_ancillary.keys()):
                                method_cmd_create_folder = method_command_ancillary['create_folder']
                            if method_cmd_create_folder is None:
                                log_stream.warning(' ===> Transfer mode "' + method_mode + ' needs to create remote folder.')
                                log_stream.warning(' ===> Check if the command settings are able to create a remote folder.')
                            else:
                                method_cmd_create_folder = deepcopy(method_cmd_create_folder.format(**template_command_line))

                            if 'uncompress_file' in list(method_command_ancillary.keys()):
                                method_cmd_uncompress_file = method_command_ancillary['uncompress_file']
                            if method_cmd_uncompress_file is None:
                                log_stream.warning(' ===> Transfer mode "' + method_mode + ' will not uncompress file.')
                                log_stream.warning(' ===> Check if the file must be compressed or not.')
                            else:
                                method_cmd_uncompress_file = deepcopy(method_cmd_uncompress_file.format(**template_command_line))

                            if 'find_file' in list(method_command_ancillary.keys()):
                                method_cmd_find_file = method_command_ancillary['find_file']
                            if method_cmd_find_file is None:
                                log_stream.warning(' ===> Transfer mode "' + method_mode + ' will not uncompress file.')
                                log_stream.warning(' ===> Check if the file must be compressed or not.')
                            else:
                                method_cmd_find_file = deepcopy(method_cmd_find_file.format(**template_command_line))

                            if 'remove_file' in list(method_command_ancillary.keys()):
                                method_cmd_remove_file = method_command_ancillary['remove_file']
                            if method_cmd_remove_file is None:
                                log_stream.warning(' ===> Transfer mode "' + method_mode + ' will not uncompress file.')
                                log_stream.warning(' ===> Check if the file must be compressed or not.')
                            else:
                                method_cmd_remove_file = deepcopy(method_cmd_remove_file.format(**template_command_line))

                        else:
                            log_stream.error(' ===> Transfer mode "' + method_mode + '" is unknown')
                            raise NotImplementedError('Case not implemented yet')

                        method_cmd_transfer_exec = deepcopy(method_command_exec.format(**template_command_line))
                        method_cmd_transfer_command = method_command_line.format(**template_command_line)

                        if file_method_src == 'ftp':
                            method_cmd_transfer = method_cmd_transfer_exec + ' "' + method_cmd_transfer_command + '"'
                        elif file_method_src == 'rsync':
                            method_cmd_transfer = method_cmd_transfer_exec + ' ' + method_cmd_transfer_command
                        else:
                            method_cmd_transfer = method_cmd_transfer_exec + ' ' + method_cmd_transfer_command

                        # Transfer file from local to remote
                        log_stream.info(
                            adjust_comment(' ------> Transfer source datasets "' + file_name_src_step +
                                           '" to destination datasets "' + file_name_dst_step + '" ... '))

                        # Check datasets source
                        if (method_mode == 'local2local') or (method_mode == 'local2remote'):
                            if os.path.exists(file_path_src_step):
                                check_dataset_source = True
                            else:
                                check_dataset_source = False
                        elif method_mode == 'remote2local':
                            check_dataset_source = True
                        else:
                            log_stream.error(' ===> Transfer mode "' + method_mode + '" is unknown')
                            raise NotImplementedError('Case not implemented yet')

                        # Condition to transfer datasets
                        if check_dataset_source:

                            # Execute command to create remote folder
                            if method_cmd_create_folder is not None:
                                execute_command(
                                    method_cmd_create_folder, command_prefix=' -------> ',
                                    command_type='Create remote folder')

                            # Execute command to transfer datasets
                            execute_command(
                                method_cmd_transfer, command_prefix=' -------> ',
                                command_type='Transfer source datasets "' +
                                             file_name_src_step + '" to destination datasets "' +
                                             file_name_dst_step + '"')

                            # Execute command to uncompress datasets
                            if method_cmd_uncompress_file is not None:
                                execute_command(
                                    method_cmd_uncompress_file, command_prefix=' -------> ',
                                    command_type='Extract compressed destination datasets "' + file_name_dst_step + '"')

                            # Execute command to remove datasets
                            if method_cmd_remove_file is not None:

                                command_find_file_code = execute_command(
                                    method_cmd_find_file, command_prefix=' -------> ',
                                    command_type='Find compressed destination datasets "' + file_name_dst_step + '"')

                                if command_find_file_code:
                                    execute_command(
                                        method_cmd_remove_file, command_prefix=' -------> ',
                                        command_type='Remove compressed destination datasets "' + file_name_dst_step + '"')

                            # Transfer datasets from local to remote
                            log_stream.info(
                                adjust_comment(' ------> Transfer source datasets "' + file_name_src_step +
                                               '" to destination datasets "' + file_name_dst_step + '" ... DONE'))

                        else:
                            log_stream.info(
                                adjust_comment(' ------> Transfer source datasets "' + file_name_src_step +
                                               '" to destination datasets "' + file_name_dst_step +
                                               '" ... SKIPPED. '
                                               'File source does not exists.'))

                    # Info dataset end (done)
                    log_stream.info(' -----> Dataset "' + dset_key_step + '" ... DONE')

                else:
                    # Info dataset end (skipped)
                    log_stream.info(' -----> Dataset "' + dset_key_step + '" ... SKIPPED. File(s) source not found')

        # Info time end
        log_stream.info(' ----> Time "' + time_step.strftime(format=time_format_algorithm) + '" ... DONE')
        # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm end
    log_stream.info(' ---> Transfer datasets from source to destination location(s) ... DONE')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    alg_time_elapsed = round(time.time() - alg_time_start, 1)

    log_stream.info(' ')
    log_stream.info('[' + project_name + ' ' + alg_type + ' - ' + alg_name + ' (Version ' + alg_version +
                    ' - Release ' + alg_release + ')]')
    log_stream.info(' ==> TIME ELAPSED: ' + str(alg_time_elapsed) + ' seconds')
    log_stream.info(' ==> ... END')
    log_stream.info(' ==> Bye, Bye')
    log_stream.info(' ============================================================================ ')
    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to execute command line
def execute_command(command_line, command_prefix=' ---> ', command_type='Execute command',
                    command_st_out=subprocess.DEVNULL, command_st_err=subprocess.DEVNULL):

    # Info start
    log_stream.info(command_prefix + adjust_comment(command_type) + ' ... ')

    # call command line
    log_stream.info(command_prefix + 'Execute: "' + command_line + '"')
    command_code = subprocess.call(
        command_line, shell=True,
        stdout=command_st_out, stderr=command_st_err)

    # method_return_code = os.system(method_cmd) # old code
    if command_code == 0:
        log_stream.info(command_prefix + adjust_comment(command_type) + ' ... DONE')
        return True
    else:
        log_stream.warning(' ===> Execution return with non-zero code for the submitted command line.')
        log_stream.info(command_prefix + adjust_comment(command_type) + ' ... SKIPPED. ')
        return False

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to fill object
def fill_object(data_structure_raw, data_template=None):

    tag_dict = deepcopy(data_template)

    if isinstance(data_structure_raw, str):
        data_structure_filled = deepcopy(data_structure_raw.format(**tag_dict))
    elif isinstance(data_structure_raw, dict):
        data_structure_filled = {}
        for data_key_raw, data_value_raw in data_structure_raw.items():

            if data_key_raw is not None:
                data_key_brackets = re.sub(r'[^(){}[\]]', '', data_key_raw)
                if data_key_brackets:
                    data_key_filled = data_key_raw.format(**tag_dict)
                else:
                    data_key_filled = deepcopy(data_key_raw)
            else:
                log_stream.error(' ===> Object key is defined by NoneType')
                raise RuntimeError('Key must be defined by a defined string')

            if data_value_raw is not None:
                data_value_brackets = re.sub(r'[^(){}[\]]', '', data_value_raw)
                if data_value_brackets:
                    data_value_filled = data_value_raw.format(**tag_dict)
                else:
                    data_value_filled = deepcopy(data_value_raw)
            else:
                data_value_raw = None
            data_structure_filled[data_key_filled] = data_value_filled
    else:
        log_stream.error(' ===> Object is not supported by the filling method')
        raise NotImplementedError('Case not implemented yet')

    return data_structure_filled
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to make folder
def make_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to adjust string comment
def adjust_comment(string_comment):
    string_comment = string_comment.replace('""', '')
    string_comment = re.sub("\s\s+", " ", string_comment)
    return string_comment
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# -------------------------------------------------------------------------------------
