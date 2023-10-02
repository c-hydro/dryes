"""
DRYES Drought Metrics Tool - SPEI Standardized Precipitation Evapotranspiration Index
__date__ = '20231002'
__version__ = '1.0.1'
__author__ =
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org',
        'Michel Isabellon (michel.isabellon@cimafoundation.org',
        'Edoardo Cremonese (edoardo.cremonese@cimafoundation.org)

__library__ = 'dryes'

General command line:
python dryes_SPEI_main.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM" -time_history_start "yyyy-mm-dd HH:MM" -time_history_end  "yyyy-mm-dd HH:MM"

Version(s):
20231002 (1.0.1) --> Added mkdir in output folder
20230925 (1.0.0) --> First release
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
import scipy.stats as stat
from astropy.convolution import convolve, Gaussian2DKernel
from lmoments3 import distr

from dryes_SPEI_utils_json import read_file_json
from dryes_SPEI_geo import read_file_raster, resample_data
from dryes_SPEI_utils_time import check_end_start_month
from dryes_SPEI_utils_generic import load_monthly_avg_data_from_geotiff, fill_tags2string
from dryes_SPEI_utils_stat import fit_gev_2d
from dryes_SPEI_tiff import write_file_tiff

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'SPEI DROUGHT METRIC'
alg_version = '1.0.1'
alg_release = '2023-10-02'
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

        logging.info(' --> Loading P!')
        data_P_monthly_history_da = \
            load_monthly_avg_data_from_geotiff(da_domain_in, period_history_daily, period_history_monthly,
                                               data_settings['data']['input']['P']['folder'],
                                               data_settings['data']['input']['P']['filename'],
                                               data_settings['algorithm']['template'],
                                               data_settings['index_info']['aggregation_method'],
                                               data_settings['data']['input']['P']['check_range'],
                                               data_settings['data']['input']['P']['range'],
                                               data_settings['data']['input']['P']['check_climatology_MAX'],
                                               data_settings['data']['input']['P']['path_climatology_MAX'],
                                               data_settings['data']['input']['P']['threshold_climatology_MAX'],
                                               data_settings['data']['input']['P']['multidaily_cumulative'],
                                               data_settings['data']['input']['P']['number_days_cumulative'])

        logging.info(' --> Loading PET!')
        data_PET_monthly_history_da = \
            load_monthly_avg_data_from_geotiff(da_domain_in, period_history_daily, period_history_monthly,
                                               data_settings['data']['input']['PET']['folder'],
                                               data_settings['data']['input']['PET']['filename'],
                                               data_settings['algorithm']['template'],
                                               data_settings['index_info']['aggregation_method'],
                                               data_settings['data']['input']['PET']['check_range'],
                                               data_settings['data']['input']['PET']['range'],
                                               data_settings['data']['input']['PET']['check_climatology_MAX'],
                                               data_settings['data']['input']['PET']['path_climatology_MAX'],
                                               data_settings['data']['input']['PET']['threshold_climatology_MAX'],
                                               data_settings['data']['input']['PET']['multidaily_cumulative'],
                                               data_settings['data']['input']['PET']['number_days_cumulative'])

        # compute D (which is the actual statistics to be used in SPEI)
        data_Di_monthly_history_da = data_P_monthly_history_da - data_PET_monthly_history_da
        #for i in range(np.shape(data_Di_monthly_history_da.values)[2]):
            #plt.figure()
            #plt.imshow(data_Di_monthly_history_da.values[:,:,i])
            #plt.colorbar()
            #plt.savefig(str(i) + '_Di.png')
            #plt.close()

        # we loop on aggregation times, aggregate & compute monthly parameters
        for i_agg, agg_window in enumerate(data_settings['index_info']['aggregation_months']):

            logging.info(' --> Started parameter fitting for aggregation ' + str(agg_window))

            # We create a 3d data array & aggregate on time window
            if data_settings['index_info']['aggregation_method'] == 'mean':
                data_Di_monthly_history_agg = data_Di_monthly_history_da.rolling(time=int(agg_window)).mean()
            elif data_settings['index_info']['aggregation_method'] == 'sum':
                data_Di_monthly_history_agg = data_Di_monthly_history_da.rolling(time=int(agg_window)).sum()
            else:
                logging.error(' ===> Aggregation method not supported!')
                raise ValueError(' ===> Aggregation method not supported!')

            # loop on months, get data, compute parameters
            for month_id, ds_month in data_Di_monthly_history_agg.groupby('time.month'):

                distr_param = fit_gev_2d(ds_month.values, data_settings['index_info']['p_value_threshold'],
                                         data_settings['algorithm']['flags']['perform_kstest'],
                                         da_domain_in.values)

                logging.info(' --> Fitted parameters for aggregation ' + str(agg_window) + ' and month ' + str(month_id))

                # we export geotiff to file
                path_geotiff_parameters = data_settings['data']['outcome']['path_output_parameters']
                tag_filled = {'aggregation': str(agg_window),
                              'month': str(month_id)}
                path_geotiff_parameters = \
                    fill_tags2string(path_geotiff_parameters, data_settings['algorithm']['template'], tag_filled)
                ndarray_input = [distr_param[:,:,0], distr_param[:,:,1], distr_param[:,:,2]]
                write_file_tiff(path_geotiff_parameters, ndarray_input, wide_domain_in, high_domain_in,
                                transform_domain_in, 'EPSG:4326')
                logging.info(' --> Parameter saved to ' + path_geotiff_parameters)

        logging.info(' --> Parameter estimation ... END!')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # load current data for computing SPEI

    logging.info(' --> SPEI computation ... START!')
    
    # load P data for the current period
    logging.info(' --> Loading P for the current period!')
    data_P_monthly_current_da = \
        load_monthly_avg_data_from_geotiff(da_domain_in, period_now_daily, period_now_monthly,
                                               data_settings['data']['input']['P']['folder'],
                                               data_settings['data']['input']['P']['filename'],
                                               data_settings['algorithm']['template'],
                                               data_settings['index_info']['aggregation_method'],
                                               data_settings['data']['input']['P']['check_range'],
                                               data_settings['data']['input']['P']['range'],
                                               data_settings['data']['input']['P']['check_climatology_MAX'],
                                               data_settings['data']['input']['P']['path_climatology_MAX'],
                                               data_settings['data']['input']['P']['threshold_climatology_MAX'],
                                               data_settings['data']['input']['P']['multidaily_cumulative'],
                                               data_settings['data']['input']['P']['number_days_cumulative'])

    logging.info(' --> Loading PET for the current period!')
    data_PET_monthly_current_da = \
        load_monthly_avg_data_from_geotiff(da_domain_in, period_now_daily, period_now_monthly,
                                               data_settings['data']['input']['PET']['folder'],
                                               data_settings['data']['input']['PET']['filename'],
                                               data_settings['algorithm']['template'],
                                               data_settings['index_info']['aggregation_method'],
                                               data_settings['data']['input']['PET']['check_range'],
                                               data_settings['data']['input']['PET']['range'],
                                               data_settings['data']['input']['PET']['check_climatology_MAX'],
                                               data_settings['data']['input']['PET']['path_climatology_MAX'],
                                               data_settings['data']['input']['PET']['threshold_climatology_MAX'],
                                               data_settings['data']['input']['PET']['multidaily_cumulative'],
                                               data_settings['data']['input']['PET']['number_days_cumulative'])

    # compute D (which is the actual statistics to be used in SPEI)
    data_Di_monthly_current_da = data_P_monthly_current_da - data_PET_monthly_current_da

    # we loop on aggregation times to compute statistics
    for i_agg, agg_window in enumerate(data_settings['index_info']['aggregation_months']):
        logging.info(' --> Started estimation of SPEI for aggregation ' + str(agg_window))

        if data_settings['index_info']['aggregation_method'] == 'mean':
            data_Di_monthly_current_da_agg = data_Di_monthly_current_da.rolling(time=int(agg_window)).mean()
        elif data_settings['index_info']['aggregation_method'] == 'sum':
            data_Di_monthly_current_da_agg = data_Di_monthly_current_da.rolling(time=int(agg_window)).sum()
        else:
            logging.error(' ===> Aggregation method not supported!')
            raise ValueError(' ===> Aggregation method not supported!')

        #we take the last value (for which we want to compute SPEI)
        Di_this_month = data_Di_monthly_current_da_agg.values[:,:,-1]

        # plt.figure()
        # plt.imshow(Di_this_month)
        # plt.savefig('Di_this_month.png')
        # plt.colorbar()
        # plt.close()

        # we load parameter values for this month and this aggregation
        path_geotiff_parameters = data_settings['data']['outcome']['path_output_parameters']
        tag_filled = {'aggregation': str(agg_window),
                      'month': str(period_now_daily[-1].month)}
        path_geotiff_parameters = \
            fill_tags2string(path_geotiff_parameters, data_settings['algorithm']['template'], tag_filled)
        parameters = xr.open_rasterio(path_geotiff_parameters)
        logging.info(' --> Parameter loaded from ' + path_geotiff_parameters)

        # compute SPEI
        probVal = distr.gev.cdf(Di_this_month, c=parameters[0,:,:], loc=parameters[1,:,:], scale=parameters[2,:,:])
        probVal[probVal == 0] = 0.0000001
        probVal[probVal == 1] = 0.9999999
        SPEI = stat.norm.ppf(probVal, loc=0, scale=1)
        logging.info(' --> SPEI computed for aggregation ' + str(agg_window))

        # plt.figure()
        # plt.imshow(SPEI)
        # plt.colorbar()
        # plt.savefig('SPEI.png')
        # plt.close()
        #
        # plt.figure()
        # plt.imshow(probVal)
        # plt.colorbar()
        # plt.savefig('probVal.png')
        # plt.close()

        # resample to fill NaN
        SPEI_dframe = pd.DataFrame(columns=['data', 'lon', 'lat'])
        SPEI_dframe['data'] = SPEI.flatten()
        SPEI_dframe['lon'] = lons_in.flatten()
        SPEI_dframe['lat'] = lats_in.flatten()
        SPEI_dframe = SPEI_dframe.dropna()
        SPEI_filled = resample_data(SPEI_dframe, lons_in, lats_in,
                                    var_name_data='data', var_name_geo_x='lon',
                                    var_name_geo_y='lat',
                                    search_radius_fill_nan = data_settings['index_info']['search_radius_fill_nan'])
        SPEI_filled = SPEI_filled.values
        logging.info(' --> SPEI nan filled for aggregation ' + str(agg_window))

        # plt.figure()
        # plt.imshow(SPEI_filled)
        # plt.colorbar()
        # plt.savefig('SPEI_filled.png')
        # plt.close()

        # smoothing
        kernel = Gaussian2DKernel(x_stddev=data_settings['index_info']['stddev_kernel_smoothing'])
        SPEI_filled_smoothed = convolve(SPEI_filled, kernel)
        logging.info(' --> SPEI smoothed for aggregation ' + str(agg_window))
        # plt.figure()
        # plt.imshow(SPEI_filled_smoothed)
        # plt.colorbar()
        # plt.savefig('SPEI_filled_smoothed.png')
        # plt.close()

        # static mask before saving (if needed)
        if data_settings['algorithm']['flags']['mask_results_static']:
            SPEI_filled_smoothed[da_domain_in.values == 0] = np.nan
            SPEI_filled_smoothed[np.isnan(da_domain_in.values)] = np.nan
            logging.info(' --> SPEI masked for aggregation ' + str(agg_window))
        # plt.figure()
        # plt.imshow(SPEI_filled_smoothed)
        # plt.colorbar()
        # plt.savefig('SPEI_filled_smoothed_static_masked.png')
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
        write_file_tiff(path_geotiff_output, SPEI_filled_smoothed, wide_domain_in, high_domain_in,
                        transform_domain_in, 'EPSG:4326')
        logging.info(' --> SPEI saved at for aggregation ' + str(agg_window) + ' at ' + path_geotiff_output)

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
