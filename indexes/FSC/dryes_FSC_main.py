"""
DRYES Drought Metrics Tool - Fractional snow-cover anomaly from H12
__date__ = '20230830'
__version__ = '1.0.1'
__author__ =
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org',
        'Michel Isabellon (michel.isabellon@cimafoundation.org'

__library__ = 'dryes'

General command line:
python dryes_FSC_main.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM" -year_history_start "yyyy" -year_history_end "yyyy"

Version(s):
20230830 (1.0.1) --> Minor updates to stop computations on February 29
20230727 (1.0.0) --> First release!
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import logging
from os.path import join
from argparse import ArgumentParser
import matplotlib.pylab as plt
import pandas as pd
import xarray as xr
import numpy as np
import sys
import os
import rasterio
from time import time, strftime, gmtime
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from dryes_FSC_utils_json import read_file_json
from dryes_FSC_utils_time import set_time
from dryes_FSC_utils_geo import read_file_raster, h12_converter
from dryes_FSC_utils_generic import fill_tags2string
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'FSC ANOMALY'
alg_version = '1.0.1'
alg_release = '2023-08-30'
alg_type = 'DroughtMetrics'
# Algorithm parameter(s)
time_format_algorithm = '%Y-%m-%d %H:%M'
# -------------------------------------------------------------------------------------

# Script Main
def main():

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    [file_script, file_settings, time_arg, year_history_start, year_history_end] = get_args()

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

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    #Load output grid
    logging.info(' --> Load target grid ... ')
    da_domain_target, wide_domain_target, high_domain_target, proj_domain_target, transform_domain_target, \
        bounding_box_domain_target, no_data_domain_target, crs_domain_target, lons_target, lats_target =\
        read_file_raster(data_settings['data']['outcome']['grid_out'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load target grid ... DONE')
    da_domain_target.values[da_domain_target.values == no_data_domain_target] = np.nan

    # plt.figure()
    # plt.imshow(da_domain_target.values)
    # plt.savefig('da_domain_target.png')
    # plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Iterate over time steps
    input_settings = data_settings['data']['input']
    for time_i, time_date in enumerate(time_range):

        if (time_date.month == 2) &  (time_date.day == 29) is False:
            # if it's 29/2, we skip computations (we do not support 29/2 during leap yrs)

            # load historical data for this date and n previous days according to days_moving_mean_FSC
            years_history = np.arange(int(year_history_start), int(year_history_end) + 1, 1)
            h12_history = np.empty(shape= (high_domain_target, wide_domain_target, len(years_history)))*np.nan

            for year_i, year in enumerate(years_history):

                logging.info(' --> Load historical h12 data for year ...' + str(year))

                time_now_pd = pd.to_datetime(time_date).round("D")
                time_this_year_pd = time_now_pd.replace(year=int(year))
                period_this_year = pd.date_range(end= time_this_year_pd,
                                                 periods= data_settings['index_info']['days_moving_mean_FSC'])

                h12_this_yr_cum = np.empty(shape=(high_domain_target, wide_domain_target, period_this_year.__len__())) * np.nan
                for time_i, time_this_day in enumerate(period_this_year):

                    logging.info(' --> Load historical h12 data for day ...' + time_this_day.strftime("%Y/%m/%d"))

                    path_file_tiff = os.path.join(data_settings['data']['input']['folder_FSC_resampled'],
                                             data_settings['data']['input']['filename_FSC_resampled'])
                    tag_filled = {'outcome_sub_path_time': time_this_day,
                                  'outcome_datetime': time_this_day}
                    path_file_tiff = fill_tags2string(path_file_tiff, data_settings['algorithm']['template'], tag_filled)

                    path_file_grib = os.path.join(data_settings['data']['input']['folder'],
                                                  data_settings['data']['input']['filename'])
                    tag_filled = {'source_gridded_sub_path_time': time_this_day,
                                  'source_datetime': time_this_day}
                    path_file_grib = fill_tags2string(path_file_grib, data_settings['algorithm']['template'], tag_filled)

                    if os.path.isfile(path_file_tiff):

                        logging.info(' --> Loading from resampled tif at ' + path_file_tiff)
                        h12_this_day = xr.open_rasterio(path_file_tiff)
                        h12_this_day = h12_this_day.values
                        h12_this_day = np.squeeze(h12_this_day)

                    elif os.path.isfile(path_file_grib):

                        logging.info(' --> Loading from grib at ' + path_file_grib)

                        h12_this_day = h12_converter(path_file_grib, path_file_tiff, da_domain_target, crs_domain_target,
                                                     transform_domain_target,
                                                     data_settings['algorithm']['general']['path_cdo'],
                                                     input_settings['layer_data_name'],
                                                     input_settings['layer_data_lon'],
                                                     input_settings['layer_data_lat'],
                                                     input_settings['lat_lon_scale_factor'],
                                                     input_settings['valid_range'])
                    else:
                        logging.warning(' --> grib file not available at ' + path_file_grib)
                        h12_this_day = np.empty(shape=(high_domain_target, wide_domain_target)) * np.nan

                    h12_this_yr_cum[:, :, time_i] = h12_this_day

                h12_this_yr_avg = np.nanmean(h12_this_yr_cum, axis=2)
                h12_history[:, :, year_i] = h12_this_yr_avg

            #now load data for this year ...
            period_current_year = pd.date_range(end=time_now_pd,
                                                periods=data_settings['index_info']['days_moving_mean_FSC'])
            h12_current_yr_cum = np.empty(shape=(high_domain_target, wide_domain_target,
                                                 period_current_year.__len__())) * np.nan
            for time_i, time_this_day in enumerate(period_current_year):
                logging.info(' --> Load current-yr h12 data for day ...' + time_this_day.strftime("%Y/%m/%d"))

                path_file_tiff = os.path.join(data_settings['data']['input']['folder_FSC_resampled'],
                                              data_settings['data']['input']['filename_FSC_resampled'])
                tag_filled = {'outcome_sub_path_time': time_this_day,
                              'outcome_datetime': time_this_day}
                path_file_tiff = fill_tags2string(path_file_tiff, data_settings['algorithm']['template'], tag_filled)

                path_file_grib = os.path.join(data_settings['data']['input']['folder'],
                                              data_settings['data']['input']['filename'])
                tag_filled = {'source_gridded_sub_path_time': time_this_day,
                              'source_datetime': time_this_day}
                path_file_grib = fill_tags2string(path_file_grib, data_settings['algorithm']['template'], tag_filled)

                if os.path.isfile(path_file_tiff):

                    logging.info(' --> Loading from resampled tif at ' + path_file_tiff)

                    h12_this_day = xr.open_rasterio(path_file_tiff)
                    h12_this_day = h12_this_day.values
                    h12_this_day = np.squeeze(h12_this_day)

                elif os.path.isfile(path_file_grib):

                    logging.info(' --> Loading from grib at ' + path_file_grib)

                    h12_this_day = h12_converter(path_file_grib, path_file_tiff, da_domain_target, crs_domain_target,
                                                 transform_domain_target,
                                                 data_settings['algorithm']['general']['path_cdo'],
                                                 input_settings['layer_data_name'],
                                                 input_settings['layer_data_lon'],
                                                 input_settings['layer_data_lat'],
                                                 input_settings['lat_lon_scale_factor'],
                                                 input_settings['valid_range'])
                else:
                    logging.warning(' --> grib file not available at ' + path_file_grib)
                    h12_this_day = np.empty(shape=(high_domain_target, wide_domain_target)) * np.nan

                h12_current_yr_cum[:, :, time_i] = h12_this_day

            h12_current_yr_avg = np.nanmean(h12_current_yr_cum, axis=2)

            # compute anomaly
            logging.info(' --> Computing anomaly for day ... ' + time_date.strftime("%Y/%m/%d"))

            h12_history_avg = np.nanmean(h12_history, axis=2)
            anomaly_tmp = (h12_current_yr_avg - h12_history_avg)/h12_history_avg*100
            anomaly_tmp[anomaly_tmp <= data_settings['data']['outcome']['limits_anomaly_output'][0]] = np.nan
            anomaly_tmp[anomaly_tmp >= data_settings['data']['outcome']['limits_anomaly_output'][1]] = np.nan

            # save geotiff
            path_output_tiff = os.path.join(data_settings['data']['outcome']['path_output_results'],
                                          data_settings['data']['outcome']['filename_output_results'])
            tag_filled = {'outcome_sub_path_time': time_date,
                          'outcome_datetime': time_date}
            path_output_tiff = fill_tags2string(path_output_tiff, data_settings['algorithm']['template'], tag_filled)
            dir_output_tiff, name_output_tiff = os.path.split(path_output_tiff)
            if os.path.isdir(dir_output_tiff) is False:
                os.makedirs(dir_output_tiff)
            layer_out = anomaly_tmp.astype(np.float32)
            with rasterio.open(path_output_tiff, 'w', height=da_domain_target.shape[0],
                               width=da_domain_target.shape[1], count=1, dtype='float32',
                               crs=crs_domain_target, transform=transform_domain_target, driver='GTiff',
                               nodata=-9999,
                               compress='lzw') as out:
                out.write(layer_out, 1)
            logging.info(' --> Anomaly saved at ' + path_output_tiff)

        else:
            logging.warning(' --> Computation for ' + time_date.strftime("%Y-%m-%d %H:%M")
                            + ' SKIPPED to avoid spurious results')

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
    parser_handle.add_argument('-year_history_start', action="store", dest="alg_year_history_start")
    parser_handle.add_argument('-year_history_end', action="store", dest="year_history_end")
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

    if parser_values.alg_year_history_start:
        alg_year_history_start = parser_values.alg_year_history_start
    else:
        alg_year_history_start = None

    if parser_values.year_history_end:
        year_history_end = parser_values.year_history_end
    else:
        year_history_end = None

    return alg_script, alg_settings, alg_time_now, alg_year_history_start, year_history_end

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