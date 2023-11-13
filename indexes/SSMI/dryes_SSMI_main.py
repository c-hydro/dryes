"""
DRYES Drought Metrics Tool - SSMI Standardized Soil Moisture Index
__date__ = '20230922'
__version__ = '1.1.1'
__author__ =
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org',
        'Michel Isabellon (michel.isabellon@cimafoundation.org'

__library__ = 'dryes'

General command line:
python dryes_SSMI_main.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM" -time_history_start "yyyy-mm-dd HH:MM" -time_history_end  "yyyy-mm-dd HH:MM"

Version(s):
20230621 (1.1.1) --> Modified output path for maps to include yy/mm/dd folders (if needed)
20230621 (1.1.0) --> Added dynamic masking (e.g., using SWE) and additional minor changes to code
20230621 (1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Library
import logging
from os.path import join
from argparse import ArgumentParser
import pandas as pd
import xarray as xr
import rioxarray
import numpy as np
import sys
import os
import matplotlib.pylab as plt
from time import time, strftime, gmtime
import matplotlib as mpl
import scipy.stats as stat
from astropy.convolution import convolve, Gaussian2DKernel

from dryes_SSMI_utils_json import read_file_json
from dryes_SSMI_utils_generic import fill_tags2string, create_darray_3d, load_monthly_avg_data_from_geotiff
from dryes_SSMI_geo import read_file_raster, resample_data
from dryes_SSMI_utils_time import check_end_start_month
from dryes_SSMI_utils_stat import kstest_2d_beta
from dryes_SSMI_tiff import write_file_tiff

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'SSMI DROUGHT METRIC'
alg_version = '1.1.1'
alg_release = '2023-09-18'
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
                                               data_settings['data']['input']['folder'],
                                               data_settings['data']['input']['filename'],
                                               data_settings['algorithm']['template'],
                                               data_settings['index_info']['aggregation_method'],
                                               data_settings['data']['input']['check_range'],
                                               data_settings['data']['input']['range'])

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

                data_mean = ds_month.mean(dim='time', skipna=True, keep_attrs=True)
                data_var = ds_month.var(dim='time', skipna=True, keep_attrs=True)

                """
                ra = data_max - data_min
                mu = (data_mean - data_min)/ra
                va = data_var/ra/ra
                ab = mu*(1 - mu)/va - 1
                fit_alpha = mu*ab
                fit_beta = (1 - mu)*ab
                """

                fit_alpha = (data_mean.values * data_mean.values) * (1 - data_mean.values) / data_var.values - data_mean.values
                fit_beta = (data_mean.values * (1 - data_mean.values) / data_var.values - 1) * (1 - data_mean.values)

                # plt.figure()
                # plt.imshow(fit_alpha)
                # plt.colorbar()
                # plt.savefig('alfa_month_' + str(i_agg) + '_agg_' + str(agg_window) + '.png')
                # plt.close()
                #
                # plt.figure()
                # plt.imshow(fit_beta)
                # plt.colorbar()
                # plt.savefig('beta_month_' + str(i_agg) + '_agg_' + str(agg_window) + '.png')
                # plt.close()

                distr_param = np.zeros(shape=(high_domain_in, wide_domain_in, 6))*np.nan
                distr_param[:, :, 0] = fit_alpha
                distr_param[:, :, 1] = fit_beta
                distr_param[:, :, 2] = np.zeros_like(fit_beta)
                distr_param[:, :, 3] = np.zeros_like(fit_beta) + 1
                distr_param[:, :, 4] = data_mean
                distr_param[:, :, 5] = ds_month.std(dim='time', skipna=True, keep_attrs=True).values

                logging.info(' --> Fitted parameters for aggregation ' + str(agg_window) + ' and month ' + str(month_id))

                if data_settings['algorithm']['flags']['perform_kstest']:
                     logging.info(' --> Applying KS for aggregation ' + str(agg_window) + ' and month ' + str(month_id))
                     # we perform the KS test and set to nan pixels with unsupported distribution
                     distr_param = kstest_2d_beta(ds_month.values, distr_param,
                                                  data_settings['index_info']['p_value_threshold'])

                # we export geotiff to file
                path_geotiff_parameters = data_settings['data']['outcome']['path_output_parameters']
                tag_filled = {'aggregation': str(agg_window),
                              'month': str(month_id)}
                path_geotiff_parameters = \
                    fill_tags2string(path_geotiff_parameters, data_settings['algorithm']['template'], tag_filled)
                ndarray_input = [distr_param[:,:,0], distr_param[:,:,1], distr_param[:,:,2],
                                 distr_param[:,:,3], distr_param[:,:,4], distr_param[:,:,5]]
                write_file_tiff(path_geotiff_parameters, ndarray_input, wide_domain_in, high_domain_in,
                                transform_domain_in, 'EPSG:4326')
                logging.info(' --> Parameter saved to ' + path_geotiff_parameters)

        logging.info(' --> Parameter estimation ... END!')

    # load current data for computing SSMI
    logging.info(' --> SSMI computation ... START!')

    data_month_values_now_ALL_da =  load_monthly_avg_data_from_geotiff(da_domain_in, period_now_daily, period_now_monthly,
                                                                       data_settings['data']['input']['folder'],
                                                                       data_settings['data']['input']['filename'],
                                                                       data_settings['algorithm']['template'],
                                                                       data_settings['index_info'][
                                                                           'aggregation_method'],
                                                                       data_settings['data']['input']['check_range'],
                                                                       data_settings['data']['input']['range'])

    # if dynamic masking is activated, then load data for that
    if data_settings['algorithm']['flags']['mask_results_dynamic']:

        logging.info(' --> Collecting dynamic mask data .. START!')

        data_dynamic_mask_da = \
            load_monthly_avg_data_from_geotiff(da_domain_in, period_now_daily,
                                               period_now_monthly, data_settings['data']['input']['dynamic_mask_settings']['folder'],
                                               data_settings['data']['input']['dynamic_mask_settings']['filename'],
                                               data_settings['algorithm']['template'],
                                               data_settings['data']['input']['dynamic_mask_settings']['aggregation_method'],
                                               data_settings['data']['input']['dynamic_mask_settings']['check_range'],
                                               data_settings['data']['input']['dynamic_mask_settings']['range_mask'])

        logging.info(' --> Collecting dynamic mask data .. DONE!')

    # we loop on aggregation times to compute statistics
    for i_agg, agg_window in enumerate(data_settings['index_info']['aggregation_months']):
        logging.info(' --> Started estimation of SSMI for aggregation ' + str(agg_window))

        if data_settings['index_info']['aggregation_method'] == 'mean':
            data_month_values_now_ALL_agg = data_month_values_now_ALL_da.rolling(time=int(agg_window)).mean()
            if data_settings['algorithm']['flags']['mask_results_dynamic']:
                data_dynamic_mask_da_agg = data_dynamic_mask_da.rolling(time=int(agg_window)).mean()
        elif data_settings['index_info']['aggregation_method'] == 'sum':
            data_month_values_now_ALL_agg = data_month_values_now_ALL_da.rolling(time=int(agg_window)).sum()
            if data_settings['algorithm']['flags']['mask_results_dynamic']:
                data_dynamic_mask_da_agg = data_dynamic_mask_da.rolling(time=int(agg_window)).sum()
        else:
            logging.error(' ===> Aggregation method not supported!')
            raise ValueError(' ===> Aggregation method not supported!')

        #we take the last value (for which we want to compute SSMI)
        data_this_month = data_month_values_now_ALL_agg.values[:,:,-1]
        if data_settings['algorithm']['flags']['mask_results_dynamic']:
            data_dynamic_mask_da_agg_this_month = data_dynamic_mask_da_agg.values[:,:,-1]

        # we load parameter values for this month and this aggregation
        path_geotiff_parameters = data_settings['data']['outcome']['path_output_parameters']
        tag_filled = {'aggregation': str(agg_window),
                      'month': str(period_now_daily[-1].month)}
        path_geotiff_parameters = \
            fill_tags2string(path_geotiff_parameters, data_settings['algorithm']['template'], tag_filled)
        parameters = rioxarray.open_rasterio(path_geotiff_parameters)
        logging.info(' --> Parameter loaded from ' + path_geotiff_parameters)

        # compute SSMI
        probVal = stat.beta.cdf(data_this_month, a=parameters[0,:,:], b=parameters[1,:,:],
                                loc=parameters[2,:,:], scale=parameters[3,:,:])
        probVal[probVal == 0] = 0.0000001
        probVal[probVal == 1] = 0.9999999
        SSMI = stat.norm.ppf(probVal, loc=0, scale=1)
        logging.info(' --> SSMI computed for aggregation ' + str(agg_window))

        # plt.figure()
        # plt.imshow(SSMI)
        # plt.colorbar()
        # plt.savefig('SSMI.png')
        # plt.close()

        # resample to fill NaN
        SSMI_dframe = pd.DataFrame(columns=['data', 'lon', 'lat'])
        SSMI_dframe['data'] = SSMI.flatten()
        SSMI_dframe['lon'] = lons_in.flatten()
        SSMI_dframe['lat'] = lats_in.flatten()
        SSMI_dframe = SSMI_dframe.dropna()
        SSMI_filled = resample_data(SSMI_dframe, lons_in, lats_in,
                                    var_name_data='data', var_name_geo_x='lon',
                                    var_name_geo_y='lat',
                                    search_radius_fill_nan = data_settings['index_info']['search_radius_fill_nan'])
        SSMI_filled = SSMI_filled.values
        logging.info(' --> SSMI nan filled for aggregation ' + str(agg_window))

        # plt.figure()
        # plt.imshow(SSMI_filled)
        # plt.colorbar()
        # plt.savefig('SSMI_filled.png')
        # plt.close()

        # smoothing
        kernel = Gaussian2DKernel(x_stddev=data_settings['index_info']['stddev_kernel_smoothing'])
        SSMI_filled_smoothed = convolve(SSMI_filled, kernel)
        logging.info(' --> SSMI smoothed for aggregation ' + str(agg_window))
        # plt.figure()
        # plt.imshow(SSMI_filled_smoothed)
        # plt.colorbar()
        # plt.savefig('SSMI_filled_smoothed.png')
        # plt.close()

        # dynamic mask
        if data_settings['algorithm']['flags']['mask_results_dynamic']:
            threshold = data_settings['data']['input']['dynamic_mask_settings']['threshold_0_1']
            data_dynamic_mask_da_agg_this_month[data_dynamic_mask_da_agg_this_month > threshold] = np.nan
            data_dynamic_mask_da_agg_this_month[data_dynamic_mask_da_agg_this_month < threshold] = 1
            SSMI_filled_smoothed = SSMI_filled_smoothed*data_dynamic_mask_da_agg_this_month
        # plt.figure()
        # plt.imshow(SSMI_filled_smoothed)
        # plt.colorbar()
        # plt.savefig('SSMI_filled_smoothed_dynamic_masked.png')
        # plt.close()

        # static mask before saving (if needed)
        if data_settings['algorithm']['flags']['mask_results_static']:
            SSMI_filled_smoothed[da_domain_in.values == 0] = np.nan
            SSMI_filled_smoothed[np.isnan(da_domain_in.values)] = np.nan
            logging.info(' --> SSMI masked for aggregation ' + str(agg_window))
        # plt.figure()
        # plt.imshow(SSMI_filled_smoothed)
        # plt.colorbar()
        # plt.savefig('SSMI_filled_smoothed_static_masked.png')
        # plt.close()

        # export to tiff
        path_geotiff_output = data_settings['data']['outcome']['path_output_results']
        tag_filled = {'aggregation': str(agg_window),
                      'outcome_datetime': pd.to_datetime(time_arg_now),
                      'outcome_sub_path_time': pd.to_datetime(time_arg_now)}
        path_geotiff_output = \
            fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
        dir, filename = os.path.split(path_geotiff_output)
        if os.path.isdir(dir) is False:
            os.makedirs(dir)
        write_file_tiff(path_geotiff_output, SSMI_filled_smoothed, wide_domain_in, high_domain_in,
                        transform_domain_in, 'EPSG:4326')
        logging.info(' --> SSMI saved at for aggregation ' + str(agg_window) + ' at ' + path_geotiff_output)


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
