"""
DRYES Drought Metrics Tool - Temperature anomaly
__date__ = '20231003'
__version__ = '1.0.1'
__author__ =
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org',
        'Michel Isabellon (michel.isabellon@cimafoundation.org'

__library__ = 'dryes'

General command line:
python dryes_Tanomaly_main.py -settings_file "dryes_Tanomaly.json" -time_now "yyyy-mm-dd HH:MM" -time_history_start "yyyy-mm-dd HH:MM" -time_history_end  "yyyy-mm-dd HH:MM"

Version(s):
20231003 (1.0.1) --> Added mkdir in output file (if needed)
20230831 (1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Library
import logging
from os.path import join
from argparse import ArgumentParser
import pandas as pd
import xarray as xr
import numpy as np
import sys
import os
import matplotlib.pylab as plt
from time import time, strftime, gmtime
import matplotlib as mpl
from astropy.convolution import convolve, Gaussian2DKernel

from dryes_Tanomaly_utils_json import read_file_json
from dryes_Tanomaly_geo import read_file_raster
from dryes_Tanomaly_utils_time import check_end_start_month
from dryes_Tanomaly_utils_generic import load_monthly_avg_data_from_geotiff, fill_tags2string
from dryes_Tanomaly_tiff import write_file_tiff

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'T ANOMALY DROUGHT METRIC'
alg_version = '1.0.1'
alg_release = '2023-10-03'
alg_type = 'DroughtMetrics'
# Algorithm parameter(s)
time_format_algorithm = '%Y-%m-%d %H:%M'
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Script Main
def main():

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    [file_script, file_settings, time_arg_now, time_arg_history_start, time_arg_history_end] = get_args()

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
    logging.info('[' + alg_project + '] Reference Time: ' + time_arg_now + ' GMT')
    logging.info('[' + alg_project + '] Historical Period: ' + time_arg_history_start + ' to ' + time_arg_history_start + ' GMT')
    logging.info('[' + alg_project + '] Start Program ... ')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load target grid
    logging.info(' --> Load target grid ... ')
    da_domain_in, wide_domain_in, high_domain_in, proj_domain_in, transform_domain_in, \
        bounding_box_domain_in, no_data_domain_in, crs_domain_in, lons_in, lats_in, res_in =\
        read_file_raster(data_settings['data']['input']['input_grid'])
    logging.info(' --> Load target grid ... DONE')

    # plt.figure()
    # plt.imshow(da_domain_in.values)
    # plt.savefig('da_domain_in.png')
    # plt.close()

    # -------------------------------------------------------------------------------------
    # Time algorithm information
    start_time = time()

    # we check temporal bounds and enforce start or end of month wherever needed
    time_arg_now = check_end_start_month(time_arg_now, start_month=True)
    time_arg_history_start = check_end_start_month(time_arg_history_start, start_month=True)
    time_arg_history_end = check_end_start_month(time_arg_history_end, end_month=True)

    # we build pandas datetimeindex objects for historical period
    period_history_hourly = pd.date_range(time_arg_history_start, time_arg_history_end, freq="H") # for loading data
    period_history_monthly = pd.date_range(time_arg_history_start, time_arg_history_end, freq="M") # for final aggregation

    # we build pandas datetimeindex objects for now period
    max_offset_months = np.max(data_settings['index_info']['aggregation_months'])
    time_arg_now_pd = pd.to_datetime(time_arg_now)
    period_now_hourly = pd.date_range(time_arg_now_pd - pd.DateOffset(months=max_offset_months),
                                     pd.to_datetime(time_arg_now) - pd.DateOffset(hours=1), freq="H")
    period_now_monthly = pd.date_range(time_arg_now_pd - pd.DateOffset(months=max_offset_months),
                                     pd.to_datetime(time_arg_now) - pd.DateOffset(hours=1), freq="M")
    # period_now_hourly goes from max_offset_months BEFORE time_arg_now up to the day BEFORE time_arg_now (given that the
    # latter was enforced to the first day of the month. Same for period_now_monthly, but with monthly temp res

    # -------------------------------------------------------------------------------------
    # Compute long-term average T (if needed)
    if data_settings['algorithm']['flags']['compute_long_term_mean']:

        logging.info(' --> Computation of long-term mean T ... START!')

        data_month_values_ALL_da = \
            load_monthly_avg_data_from_geotiff(da_domain_in, period_history_hourly, period_history_monthly, data_settings)
        #data_month_values_ALL_da is a 3D DataArray with lat x lon x months monthly avg temperatures. Months follow period_history_monthly

        # we loop on aggregation times, aggregate & save means
        for i_agg, agg_window in enumerate(data_settings['index_info']['aggregation_months']):

            logging.info(' --> Started computation of monthly means for aggregation ' + str(agg_window))

            data_month_values_ALL_agg = data_month_values_ALL_da.rolling(time=int(agg_window)).mean()
            # data_month_values_ALL_agg still includes monthly rasters like data_month_values_ALL_da,
            # but in this case each raster hosts avg T over agg_window months (so, e.g., if agg_window = 3 each layer
            # report avg T over that month and the previous two (that is, quarterly avg T). Note that there is a simplification here,
            # because this quarterly avg T is the avg of monthly avg temperatures, rather than the avg of the original underlying hourly data

            # loop on months, get data, save
            for month_id, ds_month in data_month_values_ALL_agg.groupby('time.month'):

                logging.info(' --> Month: ' + str(month_id))

                data_mean = ds_month.mean(dim='time', skipna=True, keep_attrs=True) #this is the mean of those monthly rasters
                # now we are looping on months, so that ds_month is the subset of data_month_values_ALL_agg for month month_id.
                # we thus take the mean of this new DataArray (data_mean), which is the long-term mean for month month_id
                # and aggregation period agg_window. We now save it, specifying both the reference month and the aggregation period, plus the hystorical period considered.

                # mask before saving (if needed)
                if data_settings['algorithm']['flags']['mask_results']:
                    data_mean = data_mean * da_domain_in.values
                    logging.info(' --> Long-term mean masked')

                # we export geotiff to file
                path_geotiff_long_term = data_settings['data']['outcome']['path_output_long_term_means']
                tag_filled = {'aggregation': str(agg_window),
                              'month': str(month_id),
                              'history_start': time_arg_history_start[0:4],
                              'history_end': time_arg_history_end[0:4]}
                path_geotiff_long_term = \
                    fill_tags2string(path_geotiff_long_term, data_settings['algorithm']['template'], tag_filled)
                write_file_tiff(path_geotiff_long_term, data_mean.values, wide_domain_in, high_domain_in,
                                transform_domain_in, 'EPSG:4326')
                logging.info(' --> Long term mean saved to ' + path_geotiff_long_term)

        logging.info(' --> Computation of long-term mean T ... END!')

    else:
        logging.info(' --> Computation of long-term mean T ... SKIPPED!')

    # load current data for computing T anomaly
    logging.info(' --> Temperature anomaly computation ... START!')

    data_month_values_now_ALL_da =  load_monthly_avg_data_from_geotiff(da_domain_in, period_now_hourly, period_now_monthly, data_settings)

    # we loop on aggregation times to compute statistics
    for i_agg, agg_window in enumerate(data_settings['index_info']['aggregation_months']):
        logging.info(' --> Started estimation of T anomaly for aggregation ' + str(agg_window))

        data_month_values_now_ALL_agg = data_month_values_now_ALL_da.rolling(time=int(agg_window)).mean()
        # like for historical data, we compute a moving mean over time using agg_window as the moving mean window.
        # each layer in data_month_values_now_ALL_agg hosts avg T over agg_window months(so, e.g., if agg_window = 3 each
        #layer reports avg T over that month and the previous two (that is, quarterly avg T). Note that there is a simplification here,
        # because this quarterly avg T is the avg of monthly avg temperatures, rather than the avg of the original underlying hourly data

        #we now take the last value (for which we want to compute anomaly)
        data_this_month = data_month_values_now_ALL_agg.values[:,:,-1]

        # we load long term anomaly for this month, this aggregation, and this historical period
        path_geotiff_long_term = data_settings['data']['outcome']['path_output_long_term_means']
        tag_filled = {'aggregation': str(agg_window),
                      'month': str(period_now_hourly[-1].month),
                      'history_start': time_arg_history_start[0:4],
                      'history_end': time_arg_history_end[0:4]}
        path_geotiff_long_term = \
            fill_tags2string(path_geotiff_long_term, data_settings['algorithm']['template'], tag_filled)
        long_term_mean = xr.open_rasterio(path_geotiff_long_term)
        long_term_mean = np.squeeze(long_term_mean.values)
        logging.info(' --> Long-term mean loaded from ' + path_geotiff_long_term)

        # compute anomaly
        anomaly = data_this_month - long_term_mean
        logging.info(' --> Anomaly computed for aggregation ' + str(agg_window))

        # smoothing
        kernel = Gaussian2DKernel(data_settings['index_info']['stddev_kernel_smoothing'])
        anomaly_smoothed = convolve(anomaly, kernel)
        logging.info(' --> Anomaly smoothed for aggregation ' + str(agg_window))

        # mask before saving (if needed)
        if data_settings['algorithm']['flags']['mask_results']:
            anomaly_smoothed = anomaly_smoothed*da_domain_in.values
            logging.info(' --> Anomaly masked for aggregation ' + str(agg_window))

        # export to tiff
        path_geotiff_output = data_settings['data']['outcome']['path_output_results']
        tag_filled = {'aggregation': str(agg_window),
                      'outcome_datetime': pd.to_datetime(time_arg_now),
                      'outcome_sub_path_time': pd.to_datetime(time_arg_now),
                      'history_start': time_arg_history_start[0:4],
                      'history_end': time_arg_history_end[0:4]}
        path_geotiff_output = \
            fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
        dir, filename = os.path.split(path_geotiff_output)
        if os.path.isdir(dir) is False:
            os.makedirs(dir)
        write_file_tiff(path_geotiff_output, anomaly_smoothed, wide_domain_in, high_domain_in,
                        transform_domain_in, 'EPSG:4326')
        logging.info(' --> Anomaly saved for aggregation ' + str(agg_window) + ' at ' + path_geotiff_output)


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
    parser_handle.add_argument('-time_history_start', action="store", dest="alg_time_history_start")
    parser_handle.add_argument('-time_history_end', action="store", dest="alg_time_history_end")
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

    if parser_values.alg_time_history_start:
        alg_time_history_start = parser_values.alg_time_history_start
    else:
        alg_time_history_start = None

    if parser_values.alg_time_history_end:
        alg_time_history_end = parser_values.alg_time_history_end
    else:
        alg_time_history_end = None

    return alg_script, alg_settings, alg_time_now, alg_time_history_start, alg_time_history_end

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
