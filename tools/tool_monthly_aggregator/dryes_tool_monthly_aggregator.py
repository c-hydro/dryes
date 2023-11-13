"""
DRYES Drought Metrics Tool - Tool to aggregate daily rasters to monthly scale
__date__ = '20231017'
__version__ = '1.0.0'
__author__ =
        'Francesco Avanzi (francesco.avanzi@cimafoundation.org'),
        'Matilde Torrassa (matilde.torrassa@edu.unito.it)'
        'Fabio Delogu (fabio.delogu@cimafoundation.org'),
        'Michel Isabellon (michel.isabellon@cimafoundation.org)'

__library__ = 'dryes'

General command line:
python dryes_tool_monthly_aggregator.py -settings_file "configuration.json" -time_now "yyyy-mm-dd 00:00"

Version(s):
20231017 (1.0.0) --> First release
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

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from dryes_tool_monthly_aggregator_json import read_file_json
from dryes_tool_monthly_aggregator_time import check_end_start_month
from dryes_tool_monthly_aggregator_geo import read_file_raster
from dryes_monthly_aggregator_generic import fill_tags2string
from dryes_tool_monthly_aggregator_tiff import load_monthly_from_geotiff, write_file_tiff

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'MONTHLY AGGREGATOR'
alg_version = '1.0.0'
alg_release = '2023-10-17'
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
    time_arg_now = check_end_start_month(time_arg, start_month=True)
    time_arg_now_pd = pd.to_datetime(time_arg_now)
    period_daily = pd.date_range(time_arg_now_pd -
                                 pd.DateOffset(months=data_settings['data']['outcome']['number_months_to_compute']),
                                      time_arg_now_pd - pd.DateOffset(days=1), freq="D")
    period_monthly = pd.date_range(time_arg_now_pd -
                                   pd.DateOffset(months=data_settings['data']['outcome']['number_months_to_compute']),
                                       time_arg_now_pd - pd.DateOffset(days=1), freq="M")
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
    # load monthly data from daily data
    data_monthly = load_monthly_from_geotiff(da_domain_target, period_daily, period_monthly,
                                             data_settings['data']['input']['folder'],
                                             data_settings['data']['input']['filename'],
                                             data_settings['algorithm']['template'],
                                             data_settings['data']['outcome']['aggregation_method'],
                                             data_settings['data']['input']['check_range'],
                                             data_settings['data']['input']['range'],
                                             data_settings['data']['input']['check_climatology_MAX'],
                                             data_settings['data']['input']['path_climatology_MAX'],
                                             data_settings['data']['input']['threshold_climatology_MAX'],
                                             data_settings['data']['input']['multidaily_cumulative'],
                                             data_settings['data']['input']['number_days_cumulative'])

    # -------------------------------------------------------------------------------------
    # export monthly geotiff
    for k in range(0,data_monthly.shape[2]):

        data_this_month = data_monthly[:,:,k].values
        if data_settings['data']['outcome']['mask']:
            data_this_month[da_domain_target.values == 0] = np.nan
            data_this_month[np.isnan(da_domain_target.values)] = np.nan

        path_geotiff_output = data_settings['data']['outcome']['path']
        tag_filled = {'outcome_datetime': period_monthly[k],
                      'outcome_sub_path_time': period_monthly[k]}
        path_geotiff_output = \
            fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
        dir, filename = os.path.split(path_geotiff_output)
        if os.path.isdir(dir) is False:
            os.makedirs(dir)
        write_file_tiff(path_geotiff_output, data_monthly[:,:,k].values, wide_domain_target, high_domain_target,
                        transform_domain_target, 'EPSG:4326')
        logging.info(' --> Monthly summary saved at ' + path_geotiff_output)

    # -------------------------------------------------------------------------------------
    # Info algorithm
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
