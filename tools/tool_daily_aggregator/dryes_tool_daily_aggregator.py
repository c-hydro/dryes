"""
DRYES Drought Metrics Tool - Tool to aggregate hourly rasters to daily scale
__date__ = '20230907'
__version__ = '1.1.0'
__author__ =
        'Francesco Avanzi (francesco.avanzi@cimafoundation.org'),
        'Matilde Torrassa (matilde.torrassa@edu.unito.it)'
        'Fabio Delogu (fabio.delogu@cimafoundation.org'),
        'Michel Isabellon (michel.isabellon@cimafoundation.org)'

__library__ = 'dryes'

General command line:
python dryes_tool_daily_aggregator.py -settings_file "configuration.json" -time_now "yyyy-mm-dd 00:00"

Version(s):
20230907 (1.1.0) --> Added min and max as aggregation options + minor changes
20230831 (1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import logging
from os.path import join
from argparse import ArgumentParser
import pandas as pd
import xarray as xr
import numpy as np
import sys
import os
# import matplotlib.pylab as plt
from time import time, strftime, gmtime
import matplotlib as mpl
import warnings
import rasterio

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from dryes_tool_daily_aggregator_json import read_file_json
from dryes_tool_daily_aggregator_time import set_time
from dryes_tool_daily_aggregator_geo import read_file_raster
from dryes_daily_aggregator_generic import fill_tags2string
from dryes_tool_daily_aggregator_tiff import write_file_tiff

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'DAILY AGGREGATOR'
alg_version = '1.1.0'
alg_release = '2023-09-07'
alg_type = 'DroughtMetrics'
# Algorithm parameter(s)
time_format_algorithm = '%Y-%m-%d %H:%M'
# -------------------------------------------------------------------------------------

# Script Main
def main():

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    # [file_script, file_settings, time_arg] = get_args()
    file_settings = "dryes_tool_daily_aggregator.json"
    time_arg = "2022-07-31"

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

    #check on settings
    if data_settings['time']['time_frequency'] != 'D':
        logging.error(' ===> Output time frequency MUST be daily. Check time_frequency in JSON file')
        raise ValueError(' ===> Output time frequency MUST be daily. Check time_frequency in JSON file')

    if data_settings['time']['time_rounding'] != 'D':
        logging.error(' ===> Output time rounding MUST be daily. Check time_rounding in JSON file')
        raise ValueError(' ===> Output time rounding MUST be daily. Check time_rounding in JSON file')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    #Load grid
    logging.info(' --> Load target grid ... ')
    da_domain_target, wide_domain_target, high_domain_target, proj_domain_target, transform_domain_target, \
        bounding_box_domain_target, no_data_domain_target, crs_domain_target, lons_target, lats_target =\
        read_file_raster(data_settings['data']['input']['input_grid'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load target grid ... DONE')

    # plt.figure()
    # plt.imshow(da_domain_target.values)
    # plt.savefig('da_domain_in.png')
    # plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Iterate over time steps
    for time_i, time_date in enumerate(time_range):

        # we initialize containers
        data_hourly_cum = np.zeros(shape=(da_domain_target.shape[0], da_domain_target.shape[1])) * np.nan

        # loop on hourly data to load
        time_steps_hourly = pd.date_range(start=time_date, end=time_date.replace(hour=23), freq='H')
        for time_hour, hour_date in enumerate(time_steps_hourly):

            path_data = os.path.join(data_settings['data']['input']['folder'],
                                     data_settings['data']['input']['filename'])
            tag_filled = {'source_gridded_sub_path_time': hour_date,
                          'source_gridded_datetime': hour_date}
            path_data = fill_tags2string(path_data, data_settings['algorithm']['template'], tag_filled)

            if os.path.isfile(path_data):
                with rasterio.open(path_data) as src: 
                    data_this_hour = src.read()
                    data_this_hour = np.squeeze(data_this_hour)
                data_hourly_cum = np.dstack((data_hourly_cum, data_this_hour))  # we add data to container
                logging.info(' --> ' + hour_date.strftime("%Y-%m-%d %H:%M") + ' loaded from ' + path_data)

            else:
                logging.warning(' --> ' + hour_date.strftime("%Y-%m-%d %H:%M") + ' MISSING at ' + path_data)
                #do nothing

        # we compute daily aggregation # modified
        if data_settings['data']['outcome']['aggregation_method'] == 'mean':
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                data_daily = np.nanmean(data_hourly_cum, 2)
        elif data_settings['data']['outcome']['aggregation_method'] == 'sum':
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                data_daily = np.nansum(data_hourly_cum, 2)
        elif data_settings['data']['outcome']['aggregation_method'] == 'min':
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                data_daily = np.nanmin(data_hourly_cum, 2)
        elif data_settings['data']['outcome']['aggregation_method'] == 'max':
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                data_daily = np.nanmax(data_hourly_cum, 2)
        else:
            logging.error(' ===> Aggregation method can only be mean, sum, min or max!')
            raise ValueError(' ===> Aggregation method can only be mean, sum, min or max!')

        # mask
        if data_settings['data']['outcome']['mask']:
            data_daily = data_daily * da_domain_target.values
            logging.info(' --> Daily data masked!')

        # save!
        path_geotiff_output = data_settings['data']['outcome']['path']
        tag_filled = {'outcome_sub_path_time': time_date,
                      'outcome_datetime': time_date}
        path_geotiff_output = \
            fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
        # create folder if missing
        if os.path.isdir(os.path.split(path_geotiff_output)[0]) is False:
            os.makedirs(os.path.split(path_geotiff_output)[0])
        write_file_tiff(path_geotiff_output, data_daily, wide_domain_target, high_domain_target,
                        transform_domain_target, 'EPSG:4326')
        logging.info(' --> Daily data saver at ' + path_geotiff_output)

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
