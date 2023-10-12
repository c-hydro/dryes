"""
DRYES Drought Metrics Tool - Tool to convert continuous values to given classes
__date__ = '20231012'
__version__ = '1.0.0'
__author__ =
        'Francesco Avanzi (francesco.avanzi@cimafoundation.org)',
        'Fabio Delogu (fabio.delogu@cimafoundation.org)',
        'Michel Isabellon (michel.isabellon@cimafoundation.org)',
        'Edoardo Cremonese (edoardo.cremonese@cimafoundation.org)'

__library__ = 'dryes'

General command line:
python dryes_index_classifier.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM"

Version(s):
20231012 (1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import logging
from os.path import join
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys
import os
import xarray as xr
from time import time, strftime, gmtime

from dryes_tool_classifier_json import read_file_json
from dryes_tool_classifier_geo import read_file_raster
from dryes_tool_classifier_tiff import write_file_tiff
from dryes_tool_classifier_utils import fill_tags2string
from dryes_tool_classifier_time import set_time

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'INDEX CLASSIFIER'
alg_version = '1.0.0'
alg_release = '2023-10-12'
alg_type = 'DroughtMetrics'
# Algorithm parameter(s)
time_format_algorithm = '%Y-%m-%d %H:%M'
# -------------------------------------------------------------------------------------

# Script Main
def main():

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    [file_script, file_settings, time_arg] = get_args()

    # Set algorithm settings
    data_settings = read_file_json(file_settings)

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(logger_file=join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info('[' + alg_project + ' ' + alg_type + ' - ' + alg_name + ' (Version ' + alg_version + ')]')
    logging.info('[' + alg_project + '] Execution Time: ' + strftime("%Y-%m-%d %H:%M", gmtime()) + ' GMT')
    logging.info('[' + alg_project + '] Reference Time: ' + time_arg + ' GMT')
    logging.info('[' + alg_project + '] Start Program ... ')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Time algorithm information
    start_time = time()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Organize time run
    time_run, time_range, time_chunks = set_time(
        time_run_args=time_arg,
        time_run_file=data_settings['time']['time_run'],
        time_run_file_start=data_settings['time']['time_start'],
        time_run_file_end=data_settings['time']['time_end'],
        time_format=time_format_algorithm,
        time_period=data_settings['time']['time_period'],
        time_frequency=data_settings['time']['time_frequency'],
        time_rounding=data_settings['time']['time_rounding'],
        time_reverse=data_settings['time']['time_reverse']
    )

    # -------------------------------------------------------------------------------------
    # check on settings
    if data_settings['time']['time_frequency'] != 'D':
        logging.error(' ===> Output time frequency MUST be daily. Check time_frequency in JSON file')
        raise ValueError(' ===> Output time frequency MUST be daily. Check time_frequency in JSON file')

    if data_settings['time']['time_rounding'] != 'D':
        logging.error(' ===> Output time rounding MUST be daily. Check time_rounding in JSON file')
        raise ValueError(' ===> Output time rounding MUST be daily. Check time_rounding in JSON file')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load region grid and list
    logging.info(' --> Load target grid ... ')
    da_domain, wide_domain, high_domain, proj_domain, transform_domain, \
        bounding_box_domain, no_data_domain, crs_domain, lons_domain, lats_domain = \
        read_file_raster(data_settings['data']['input']['input_grid'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load target grid ... DONE')
    # plt.figure()
    # plt.imshow(da_regions_target.values)
    # plt.savefig('da_domain_in.png')
    # plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Loading thresholds and classes
    logging.info(' --> Loading target classes from json ...')
    thresholds = data_settings['classes']['thresholds']
    classes = data_settings['classes']['classes']

    logging.info(' --> Loading target classes from json ... DONE')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Iterate over time steps
    for time_i, time_date in enumerate(time_range):

        # -------------------------------------------------------------------------------------
        # Load raster
        path_data = os.path.join(data_settings['data']['input']['folder'],
                                 data_settings['data']['input']['filename'])
        tag_filled = {'source_gridded_sub_path_time': pd.to_datetime(time_date),
                      'source_gridded_datetime': pd.to_datetime(time_date)}
        path_data = fill_tags2string(path_data, data_settings['algorithm']['template'], tag_filled)

        if os.path.isfile(path_data):
            logging.info(' --> Loading raster to be aggregated at ... ' + path_data)
            da_raster, wide_raster, high_raster, proj_raster, transform_raster, \
                bounding_box_raster, no_data_raster, crs_raster, lons_raster, lats_raster = \
                read_file_raster(path_data,
                                 coord_name_x='lon', coord_name_y='lat',
                                 dim_name_x='lon', dim_name_y='lat')
            logging.info(' --> LOADED!')

        else:
            logging.warning(' --> ' + time_date.strftime("%Y-%m-%d") + ' MISSING: skip to next day ...')
            continue

        # plt.figure()
        # plt.imshow(da_raster.values)
        # plt.colorbar()
        # plt.show()

        # -------------------------------------------------------------------------------------
        # Reclassify the raster data
        da_raster_class = xr.apply_ufunc(np.digitize, da_raster, thresholds)
        logging.info(' --> Classification done!')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Assign specific values to each class
        raster_class = np.zeros(shape=(da_domain.shape[0], da_domain.shape[1])) * np.nan
        for cl_target, cl_input in zip(classes, range(1,len(classes)+1)):
            logging.info(' --> Converting index class ' + str(cl_input) + ' --> with json provided class: ' + str(cl_target))
            raster_class[da_raster_class.values == cl_input] = cl_target
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Mask data if needed
        if data_settings['algorithm']['flags']['mask_results']:
            raster_class[da_domain.values == 0] = np.nan
            raster_class[np.isnan(da_domain.values)] = np.nan
            logging.info(' --> Masking done!')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Write to file
        path_geotiff_output = os.path.join(data_settings['data']['outcome']['folder'],
                                           data_settings['data']['outcome']['filename'])
        tag_filled = {'outcome_sub_path_time': pd.to_datetime(time_date),
                      'outcome_datetime': pd.to_datetime(time_date)}
        path_geotiff_output = fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
        if os.path.isdir(os.path.split(path_geotiff_output)[0]) is False:
            os.makedirs(os.path.split(path_geotiff_output)[0])
        write_file_tiff(path_geotiff_output, raster_class, wide_domain, high_domain,
                        transform_domain, 'EPSG:4326')
        logging.info(' --> Map saved for time: ' + time_date.strftime("%Y-%m-%d") + ' at: ' + path_geotiff_output)
        # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    #Info algorithm
    time_elapsed = round(time() - start_time, 1)

    logging.info(' ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> TIME ELAPSED: ' + str(time_elapsed) + ' seconds')
    logging.info(' ==> ... END')
    logging.info(' ==> Bye, Bye')
    logging.info(' ============================================================================ ')
    sys.exit(0)
    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():

    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time_now', action="store", dest="alg_time_now")
    parser_values = parser_handle.parse_args()

    alg_script = parser_handle.prog

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time_now:
        alg_time_now = parser_values.alg_time_now
    else:
        alg_time_now = None

    return alg_script, alg_settings, alg_time_now

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to set logging information
def set_logging(logger_file='log.txt', logger_format=None):
    if logger_format is None:
        logger_format = '%(asctime)s %(name)-12s %(levelname)-8s ' \
                        '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'

    # Remove old logging file
    if os.path.exists(logger_file):
        os.remove(logger_file)

    # Set level of root debugger
    logging.root.setLevel(logging.INFO)

    # Open logging basic configuration
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode='w')

    # Set logger handle
    logger_handle_1 = logging.FileHandler(logger_file, 'w')
    logger_handle_2 = logging.StreamHandler()
    # Set logger level
    logger_handle_1.setLevel(logging.INFO)
    logger_handle_2.setLevel(logging.INFO)
    # Set logger formatter
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)
    # Add handle to logging
    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)


# -------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------