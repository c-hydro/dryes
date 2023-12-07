"""
DRYES Drought Metrics Tool - SPI Standardized Soil Moisture Index
__date__ = '20230622'
__version__ = '1.0.0'
__author__ =
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org',
        'Michel Isabellon (michel.isabellon@cimafoundation.org'

__library__ = 'dryes'

General command line:
python dryes_SPI_main.py -settings_file configuration.json -time_now "yyyy-mm-dd HH:MM" -time_history_start "yyyy-mm-dd HH:MM" -time_history_end  "yyyy-mm-dd HH:MM"
-settings_file /home/michel/workspace/python/DRYES_ZEUS/analyses/SPI/dryes_SPI.json -time_now "2023-09-26 08:00" -time_history_start "2002-01-01 01:00" -time_history_end  "2004-12-31 23:00"
Version(s):
20230621 (1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Library
from os.path import join

import logging
import pandas as pd
import xarray as xr
import rioxarray
import sys

import matplotlib.pylab as plt
from time import time, strftime, gmtime
import matplotlib as mpl
import scipy.stats as stat
from astropy.convolution import convolve, Gaussian2DKernel

from dryes_SPI_utils_json import read_file_json
from dryes_SPI_utils_generic import *
from dryes_SPI_geo import read_file_raster, resample_data
from dryes_SPI_utils_time import check_end_start_month
from dryes_SPI_utils_stat import compute_gamma
from dryes_SPI_tiff import write_file_tiff

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'SPI DROUGHT METRIC'
alg_version = '1.0.0'
alg_release = '2023-10-02'
alg_type = 'DroughtMetrics'
# Algorithm parameter(s)
time_format_algorithm = '%Y-%m-%d'
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
    logging.info('[' + alg_project + '] Historical Period: ' + time_arg_history_start + ' to ' + time_arg_history_end + ' GMT')
    logging.info('[' + alg_project + '] Start Program ... ')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load target grid
    logging.info(' --> Load target grid ... ')
    da_domain_in, wide_domain_in, high_domain_in, proj_domain_in, transform_domain_in, \
        bounding_box_domain_in, no_data_domain_in, crs_domain_in, lons_in, lats_in, res_in =\
        read_file_raster(data_settings['data']['input']['input_grid'])
    logging.info(' --> Load target grid ... DONE')

    # -------------------------------------------------------------------------------------
    # Time algorithm information
    start_time = time()

    # we check temporal bounds and enforce start or end of month wherever needed
    time_arg_now = check_end_start_month(time_arg_now, start_month=True)
    time_arg_history_start = check_end_start_month(time_arg_history_start, start_month=True)
    time_arg_history_end = check_end_start_month(time_arg_history_end, end_month=True)

    # we build pandas datetimeindex objects for historical period
    period_history_daily = pd.date_range(time_arg_history_start, time_arg_history_end, freq="D") # for loading data
    period_history_monthly = pd.date_range(time_arg_history_start, time_arg_history_end, freq="M") # for final aggregation

    # we build pandas datetimeindex objects for now period
    max_offset_months = np.max(data_settings['index_info']['aggregation_months'])
    time_arg_now_pd = pd.to_datetime(time_arg_now)
    period_now_daily = pd.date_range(time_arg_now_pd - pd.DateOffset(months=max_offset_months),
                                     pd.to_datetime(time_arg_now) - pd.DateOffset(days=1), freq="D")
    period_now_monthly = pd.date_range(time_arg_now_pd - pd.DateOffset(months=max_offset_months),
                                     pd.to_datetime(time_arg_now) - pd.DateOffset(days=1), freq="M")
    # period_now_daily goes from max_offset_months BEFORE time_arg_now up to the day BEFORE time_arg_now (given that the
    # latter was enforced to the first day of the month. Same for period_now_monthly, but with monthly temp res

    # -------------------------------------------------------------------------------------
    # Compute parameters (if needed)
    if data_settings['algorithm']['flags']['compute_parameters']:

        logging.info(' --> Parameter estimation ... START!')

        data_month_values_ALL_da = \
            load_monthly_avg_data_from_geotiff(da_domain_in, period_history_daily, period_history_monthly,
                                               folder_in=data_settings['data']['input']['folder'],
                                               filename_in=data_settings['data']['input']['filename'],
                                               template=data_settings['algorithm']['template'],
                                               aggregation_method=data_settings['index_info']['aggregation_method'],
                                               mask=data_settings['algorithm']['flags']['mask_results'],
                                               check_range=data_settings['data']['input']['check_range'],
                                               range=data_settings['data']['input']['range'],
                                               check_climatology=data_settings['data']['input']['check_climatology'],
                                               path_climatology_in=data_settings['data']['input']['path_climatology'],
                                               threshold_climatology = data_settings['data']['input']['threshold_climatology'])

        # we loop on aggregation times, aggregate & compute monthly parameters
        for i_agg, agg_window in enumerate(data_settings['index_info']['aggregation_months']):

            logging.info(' --> Started parameter fitting for aggregation ' + str(agg_window))

            # We create a 3d data array & aggregate on time window
            if data_settings['index_info']['aggregation_method'] == 'mean':
                data_month_values_ALL_agg = data_month_values_ALL_da.rolling(time=int(agg_window)).mean()
            elif data_settings['index_info']['aggregation_method'] == 'sum':
                data_month_values_ALL_agg = data_month_values_ALL_da.rolling(time=int(agg_window)).sum()
            else:
                data_month_values_ALL_agg = data_month_values_ALL_da.rolling(time=int(agg_window)).mean()
                logging.warning(' ==> Aggregation method not supported. Mean enforced!')

            # loop on months, get data, compute parameters
            for month_id, ds_month in data_month_values_ALL_agg.groupby('time.month'):

                row = ds_month.shape[0]
                col = ds_month.shape[1]

                if data_settings['algorithm']['flags']['perform_kstest']:
                    p_val_th = data_settings['index_info']['p_value_threshold']
                else:
                    p_val_th = None

                distr_param = np.zeros((row, col, 4))*np.nan

                distr_param[:, :, 0] = xr.apply_ufunc(compute_gamma, ds_month, input_core_dims=[["time"]], kwargs={'par':'a','p_val_th':p_val_th}, dask = 'allowed', vectorize = True)
                distr_param[:, :, 1] = xr.apply_ufunc(compute_gamma, ds_month, input_core_dims=[["time"]], kwargs={'par':'loc','p_val_th':p_val_th}, dask = 'allowed', vectorize = True)
                distr_param[:, :, 2] = xr.apply_ufunc(compute_gamma, ds_month, input_core_dims=[["time"]], kwargs={'par':'scale','p_val_th':p_val_th}, dask = 'allowed', vectorize = True)
                distr_param[:, :, 3] = xr.apply_ufunc(compute_gamma, ds_month, input_core_dims=[["time"]], kwargs={'par':'zero_prob','p_val_th':p_val_th}, dask = 'allowed', vectorize = True)    # zero probability

                logging.info(' --> Fitted parameters for aggregation ' + str(agg_window) + ' and month ' + str(month_id))

                # we export geotiff to file
                path_geotiff_parameters = data_settings['data']['outcome']['path_output_parameters']
                tag_filled = {'aggregation': str(agg_window),
                              'month': str(month_id)}
                path_geotiff_parameters = \
                    fill_tags2string(path_geotiff_parameters, data_settings['algorithm']['template'], tag_filled)
                ndarray_input = [distr_param[:,:,0], distr_param[:,:,1], distr_param[:,:,2],
                                 distr_param[:,:,3]]
                write_file_tiff(path_geotiff_parameters, ndarray_input, wide_domain_in, high_domain_in,
                                transform_domain_in, 'EPSG:4326')
                logging.info(' --> Parameter saved to ' + path_geotiff_parameters)

        logging.info(' --> Parameter estimation ... END!')

    # load current data for computing SPI
    logging.info(' --> SPI computation ... START!')

    data_month_values_now_ALL_da =  load_monthly_avg_data_from_geotiff(da_domain_in, period_now_daily, period_now_monthly,
                                       folder_in=data_settings['data']['input']['folder'],
                                       filename_in=data_settings['data']['input']['filename'],
                                       template=data_settings['algorithm']['template'],
                                       aggregation_method=data_settings['index_info']['aggregation_method'],
                                       mask=data_settings['algorithm']['flags']['mask_results'],
                                       check_range=data_settings['data']['input']['check_range'],
                                       range=data_settings['data']['input']['range'],
                                       check_climatology=data_settings['data']['input']['check_climatology'],
                                       path_climatology_in=data_settings['data']['input']['path_climatology'],
                                       threshold_climatology=data_settings['data']['input']['threshold_climatology'])

    # we loop on aggregation times to compute statistics
    for i_agg, agg_window in enumerate(data_settings['index_info']['aggregation_months']):
        logging.info(' --> Started estimation of SPI for aggregation ' + str(agg_window))

        if data_settings['index_info']['aggregation_method'] == 'mean':
            data_month_values_now_ALL_agg = data_month_values_now_ALL_da.rolling(time=int(agg_window)).mean()
        elif data_settings['index_info']['aggregation_method'] == 'sum':
            data_month_values_now_ALL_agg = data_month_values_now_ALL_da.rolling(time=int(agg_window)).sum()
        else:
            data_month_values_now_ALL_agg = data_month_values_now_ALL_da.rolling(time=int(agg_window)).mean()
            logging.warning(' ==> Aggregation method not supported. Mean enforced!')

        #we take the last value (for which we want to compute SPI)
        data_this_month = data_month_values_now_ALL_agg.values[:,:,-1]

        # we load parameter values for this month and this aggregation
        path_geotiff_parameters = data_settings['data']['outcome']['path_output_parameters']
        tag_filled = {'aggregation': str(agg_window),
                      'month': str(period_now_daily[-1].month)}
        path_geotiff_parameters = \
            fill_tags2string(path_geotiff_parameters, data_settings['algorithm']['template'], tag_filled)
        parameters = rioxarray.open_rasterio(path_geotiff_parameters)
        logging.info(' --> Parameter loaded from ' + path_geotiff_parameters)

        # compute SPI
        probVal = stat.gamma.cdf(data_this_month, a=parameters[0,:,:], loc=parameters[1,:,:], scale=parameters[2,:,:])
        # probVal[probVal == 0] = 0.0000001
        # probVal[probVal == 1] = 0.9999999

        prob0 = parameters[3,:,:]

        probVal = prob0.values + \
                    ((1 - prob0.values) * probVal)

        SPI = stat.norm.ppf(probVal, loc=0, scale=1)
        logging.info(' --> SPI computed for aggregation ' + str(agg_window))

        # resample to fill NaN
        SPI_dframe = pd.DataFrame(columns=['data', 'lon', 'lat'])
        SPI_dframe['data'] = SPI.flatten()
        SPI_dframe['lon'] = lons_in.flatten()
        SPI_dframe['lat'] = lats_in.flatten()
        SPI_dframe = SPI_dframe.dropna()
        SPI_filled = resample_data(SPI_dframe, lons_in, lats_in,
                                    var_name_data='data', var_name_geo_x='lon',
                                    var_name_geo_y='lat',
                                    search_radius_fill_nan = data_settings['index_info']['search_radius_fill_nan'])
        SPI_filled = SPI_filled.values
        logging.info(' --> SPI nan filled for aggregation ' + str(agg_window))

        # smoothing
        kernel = Gaussian2DKernel(x_stddev=data_settings['index_info']['stddev_kernel_smoothing'])
        SPI_filled_smoothed = convolve(SPI_filled, kernel)
        logging.info(' --> SPI smoothed for aggregation ' + str(agg_window))

        # mask before saving (if needed)
        if data_settings['algorithm']['flags']['mask_results']:
            SPI_filled_smoothed = SPI_filled_smoothed*da_domain_in.values
            logging.info(' --> SPI masked for aggregation ' + str(agg_window))

        # export to tiff
        path_geotiff_output = data_settings['data']['outcome']['path_output_results']
        tag_filled = {'aggregation': str(agg_window),
                      'domain' : data_settings['index_info']['domain'],
                      'outcome_datetime': pd.to_datetime(time_arg_now),
                      'outcome_sub_path_time': pd.to_datetime(time_arg_now)}
        path_geotiff_output = \
            fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
        dir, filename = os.path.split(path_geotiff_output)
        if os.path.isdir(dir) is False:
            os.makedirs(dir)
        write_file_tiff(path_geotiff_output, SPI_filled_smoothed, wide_domain_in, high_domain_in,
                        transform_domain_in, 'EPSG:4326')
        logging.info(' --> SPI saved at for aggregation ' + str(agg_window) + ' at ' + path_geotiff_output)


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


# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------
