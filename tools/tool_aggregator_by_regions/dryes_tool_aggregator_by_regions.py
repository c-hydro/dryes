"""
DRYES Drought Metrics Tool - Tool to aggregate rasters by regions
__date__ = '20230928'
__version__ = '1.0.1'
__author__ =
        'Francesco Avanzi (francesco.avanzi@cimafoundation.org)',
        'Fabio Delogu (fabio.delogu@cimafoundation.org)',
        'Michel Isabellon (michel.isabellon@cimafoundation.org)',
        'Edoardo Cremonese (edoardo.cremonese@cimafoundation.org)'

__library__ = 'dryes'

General command line:
python dryes_tool_aggregator_by_regions.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM"

Version(s):
20230928 (1.0.1) --> Added min, mazx and mode
20230911 (1.0.0) --> First release
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
from scipy import stats
import matplotlib.pylab as plt
from time import time, strftime, gmtime
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from dryes_tool_daily_aggregator_by_regions_json import read_file_json
from dryes_tool_daily_aggregator_by_regions_geo import read_file_raster
from dryes_daily_aggregator_by_regions_generic import fill_tags2string

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'AGGREGATOR BY REGIONS'
alg_version = '1.0.1'
alg_release = '2023-09-28'
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
    # Load region grid and list
    logging.info(' --> Load target grid ... ')
    da_regions_target, wide_regions_target, high_regions_target, proj_regions_target, transform_regions_target, \
        bounding_box_regions_target, no_data_regions_target, crs_regions_target, lons_regions, lats_regions =\
        read_file_raster(data_settings['data']['input']['input_grid'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load target grid ... DONE')

    # plt.figure()
    # plt.imshow(da_regions_target.values)
    # plt.savefig('da_domain_in.png')
    # plt.close()

    # Read list of region names and IDs
    logging.info(' --> Loading list of regions as csv file ... ')
    list_regions = pd.read_csv(data_settings['data']['input']['input_list_regions_csv'],
                               keep_default_na=False, na_values=['NaN'])
    logging.info(' --> Loading list of regions as csv file ... DONE')

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load raster
    path_data = os.path.join(data_settings['data']['input']['folder_data_to_be_aggregated'],
                             data_settings['data']['input']['filename_data_to_be_aggregated'])
    tag_filled = {'source_gridded_sub_path_time': pd.to_datetime(time_arg),
                  'source_gridded_datetime': pd.to_datetime(time_arg)}
    path_data = fill_tags2string(path_data, data_settings['algorithm']['template'], tag_filled)

    logging.info(' --> Loading raster to be aggregated at ... ' + path_data)

    da_raster, wide_raster, high_raster, proj_raster, transform_raster, \
        bounding_box_raster, no_data_raster, crs_raster, lons_raster, lats_raster = \
        read_file_raster(path_data,
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> LOADED!')

    # plt.figure()
    # plt.imshow(da_raster.values)
    # plt.savefig('da_raster.png')
    # plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Resample da_regions_target on da_raster
    coordinates = {'lat': da_raster["lat"].values,
                   'lon': da_raster["lon"].values}
    da_regions_target_resampled = da_regions_target.interp(coordinates, method='nearest')
    # plt.figure()
    # plt.imshow(da_regions_target_resampled.values)
    # plt.savefig('da_regions_target_resampled.png')
    # plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Loop on list_regions and compute statistics
    summary_by_regions = pd.DataFrame(index=(['Q1', 'Q2', 'Q3', 'Min', 'Max', 'Mode']),
                                      columns=list_regions[data_settings['data']['input']['region_names_in_csv']].values)
    for index_list_regions, row_list_regions in list_regions.iterrows():

        logging.info(' --> Starting aggregation for ... ' +
                     row_list_regions[data_settings['data']['input']['region_names_in_csv']])

        summary_by_regions.loc['Q1', row_list_regions[data_settings['data']['input']['region_names_in_csv']]] =  \
            np.nanpercentile(da_raster.values[da_regions_target_resampled.values
                                    == row_list_regions[data_settings['data']['input']['region_IDs_in_csv']]], 25)
        summary_by_regions.loc['Q2', row_list_regions[data_settings['data']['input']['region_names_in_csv']]] =  \
            np.nanpercentile(da_raster.values[da_regions_target_resampled.values
                                              == row_list_regions[data_settings['data']['input']['region_IDs_in_csv']]],
                             50)
        summary_by_regions.loc['Q3', row_list_regions[data_settings['data']['input']['region_names_in_csv']]] =  \
            np.nanpercentile(da_raster.values[da_regions_target_resampled.values
                                              == row_list_regions[data_settings['data']['input']['region_IDs_in_csv']]],
                             75)

        summary_by_regions.loc['Min', row_list_regions[data_settings['data']['input']['region_names_in_csv']]] = \
            np.nanmin(da_raster.values[da_regions_target_resampled.values
                                              == row_list_regions[data_settings['data']['input']['region_IDs_in_csv']]])

        summary_by_regions.loc['Max', row_list_regions[data_settings['data']['input']['region_names_in_csv']]] = \
            np.nanmax(da_raster.values[da_regions_target_resampled.values
                                       == row_list_regions[data_settings['data']['input']['region_IDs_in_csv']]])

        summary_by_regions.loc['Mode', row_list_regions[data_settings['data']['input']['region_names_in_csv']]] = \
            stats.mode(da_raster.values[da_regions_target_resampled.values
                                       == row_list_regions[data_settings['data']['input']['region_IDs_in_csv']]],
                       nan_policy='omit')[0][0]
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Export to csv
    path_data_out = os.path.join(data_settings['data']['outcome']['path'])
    tag_filled = {'outcome_sub_path_time': pd.to_datetime(time_arg),
                  'outcome_datetime': pd.to_datetime(time_arg)}
    path_data_out = fill_tags2string(path_data_out, data_settings['algorithm']['template'], tag_filled)

    logging.info(' --> Exporting summary csv at ... ' + path_data_out)

    path, file = os.path.split(path_data_out)
    if os.path.isdir(path) is False:
        os.makedirs(path)

    summary_by_regions.to_csv(path_data_out)

    logging.info(' --> EXPORTED!')
    # -------------------------------------------------------------------------------------

#   # -------------------------------------------------------------------------------------
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