"""
DRYES Drought Metrics Tool - Tool to spatialize in-situ air temperature measurements based on elevation and homogeneous regions
__date__ = '20230828'
__version__ = '1.0.0'
__author__ =
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org',
        'Michel Isabellon (michel.isabellon@cimafoundation.org'

__library__ = 'dryes'

General command line:
python dryes_tool_interp_Tair_elevation.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM"

Version(s):
20230828 (1.0.0) --> Development start for first release
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
import netrc
import matplotlib.pylab as plt
from time import time, strftime, gmtime
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from drops2.utils import DropsCredentials
from datetime import timedelta
from astropy.convolution import convolve, Gaussian2DKernel

from dryes_tool_interp_Tair_elevation_utils_json import read_file_json
from dryes_tool_interp_Tair_elevation_utils_time import set_time
from dryes_tool_interp_Tair_elevation_utils_geo import read_file_raster
from dryes_tool_interp_Tair_elevation_utils_drops2 import GetDrops2, QAQC_climatology
from dryes_tool_interp_Tair_elevation_utils_generic import fill_tags2string
from dryes_tool_interp_Tair_elevation_utils_interp import interp_elevation_by_regions, interp_point2grid
from dryes_tool_interp_Tair_elevation_utils_tiff import write_file_tiff
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'TEMPERATURE SPATIALIZATION'
alg_version = '1.0.0'
alg_release = '2023-08-28'
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

    # load DEM
    logging.info(' --> Load DEM ... ')
    da_DEM, wide_DEM, high_DEM, proj_DEM, transform_DEM, \
        bounding_box_DEM, no_data_DEM, crs_DEM, lons_DEM, lats_DEM =\
        read_file_raster(data_settings['data']['input']['input_dem'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load DEM ... DONE')

    # plt.figure()
    # plt.imshow(da_DEM.values)
    # plt.savefig('da_DEM.png')
    # plt.close()

    # load homogeneous regions
    logging.info(' --> Load homogeneous regions ... ')
    da_homogeneous_regions, wide_homogeneous_regions, high_homogeneous_regions, \
        proj_homogeneous_regions, transform_homogeneous_regions, \
        bounding_box_homogeneous_regions, no_data_homogeneous_regions, \
        crs_homogeneous_regions, lons_homogeneous_regions, lats_homogeneous_regions =\
        read_file_raster(data_settings['data']['input']['input_homogeneous_regions'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load homogeneous regions ... DONE')

    # plt.figure()
    # plt.imshow(da_homogeneous_regions.values)
    # plt.savefig('da_homogeneous_regions.png')
    # plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Connect to drops
    info = netrc.netrc()
    username, account, password = info.authenticators(data_settings['algorithm']['info_drops']['url'])
    logging.info(' ---> Drops2 mode selected: connecting to the server...')
    DropsCredentials.set(data_settings['algorithm']['info_drops']['url'],username, password)
    logging.info(' ---> Drops2 mode selected: connected!')

    # -------------------------------------------------------------------------------------
    # Iterate over time steps
    for time_i, time_date in enumerate(time_range):

        # -------------------------------------------------------------------------------------
        # get in-situ data for time_date
        info_drops = data_settings['algorithm']['info_drops']
        df_dati, df_stations = GetDrops2((time_date - timedelta(hours=info_drops['timedelta_spinup_extraction'])),
                            time_date, info_drops['aggregation_seconds'], info_drops['group'],
                            info_drops['sensor_class'], bounding_box_domain_target, info_drops['invalid_flags'],
                                         info_drops['ntry'], info_drops['sec_sleep'])
        # -------------------------------------------------------------------------------------

        if df_dati.empty is False:

            logging.info(
                ' --> Creating Map for time: ' + time_date.strftime("%Y/%m/%d %H:%M"))

            # -------------------------------------------------------------------------------------
            # Perform advanced QA/QC based on climatology
            logging.info(' ---> Applying climatological filter')

            info_filters = data_settings['data']['input']['filters']
            path_climatology = info_filters['path_climatologic_maps']
            tag_filled = {'source_climatology_sub_path_time': time_date}
            path_climatology = fill_tags2string(path_climatology, data_settings['algorithm']['template'], tag_filled)
            df_dati, df_stations = QAQC_climatology(df_dati, df_stations,
                                                    path_climatology,
                                                    info_filters['threshold_climatology'])
            logging.info(' ---> Applying climatological filter ... DONE!')
            # -------------------------------------------------------------------------------------

            # -------------------------------------------------------------------------------------
            # Spatialize based on homogeneous regions (and apply additional QAQC)
            logging.info(' ---> Interpolating based on homogeneous regions and elevation')
            map_t_da = interp_elevation_by_regions(df_dati, df_stations, info_filters, da_domain_target,
                                                   da_DEM, da_homogeneous_regions)
            logging.info(' ---> Interpolating based on homogeneous regions and elevation ... DONE!')
            # -------------------------------------------------------------------------------------

            # -------------------------------------------------------------------------------------
            # Compute residuals
            logging.info(' ---> Applying residuals')

            info_residuals = data_settings['data']['input']['residuals']
            lon_query = xr.DataArray(df_stations['lon'], dims="points")
            lat_query = xr.DataArray(df_stations['lat'], dims="points")
            predictions_map = map_t_da.sel(lon=lon_query, lat=lat_query, method="nearest") #we extract predictions from the map at the location of all stations
            residuals = predictions_map.values - df_dati.values #we compute residuals

            #remove NaNs
            residuals_xy = np.empty([np.shape(residuals)[1], 3])*np.nan
            residuals_xy[:, 0] = np.ravel(residuals)
            residuals_xy[:, 1] = np.ravel(df_stations['lon'])
            residuals_xy[:, 2] = np.ravel(df_stations['lat'])
            residuals_xy = residuals_xy[~np.isnan(residuals_xy).any(axis=1), :]
            map_residuals = interp_point2grid(residuals_xy[:, 0], residuals_xy[:, 1], residuals_xy[:, 2],
                                              lons_target, lats_target,
                                              interp_method=info_residuals['interp_method'], interp_radius_x=info_residuals['interp_radius_x'],
                                              interp_radius_y=info_residuals['interp_radius_y'],
                                              n_cpu=info_residuals['n_cpu'],
                                              folder_tmp=info_residuals['path_map_tmp']) #we spatialize residuals
            # plt.figure()
            # plt.imshow(map_residuals)
            # plt.colorbar()
            # plt.savefig('map_residuals.png')
            # plt.close()

            #we apply residuals
            map_residuals[np.isnan(map_residuals)] = 0 # this is needed to make sure that we do not propagate nan from the residual map
            map_t_da.values = map_t_da.values - map_residuals
            logging.info(' ---> Applying residuals ... DONE!')

            #we check residual application
            predictions_map_after_residual_redistribution = map_t_da.sel(lon=lon_query, lat=lat_query,
                                           method="nearest")  # we extract predictions from the map at the location of all stations
            residuals_after_residual_redistribution = predictions_map_after_residual_redistribution.values - df_dati.values  # we compute residuals
            logging.info(' ---> Average residual BEFORE redistribution: ' + str(np.round(np.nanmean(residuals), 4)))
            logging.info(' ---> Average residual AFTER redistribution: ' +
                         str(np.round(np.nanmean(residuals_after_residual_redistribution), 4)))
            # plt.figure()
            # plt.imshow(map_t_da.values)
            # plt.colorbar()
            # plt.savefig('map_t_da_post_residuals.png')
            # plt.close()
            # -------------------------------------------------------------------------------------

            # -------------------------------------------------------------------------------------
            # smoothing
            logging.info(' ---> Applying smoothing ... ')
            kernel = Gaussian2DKernel(x_stddev=info_residuals['stddev_kernel_smoothing'])
            map_t_da.values = convolve(map_t_da.values, kernel)
            logging.info(' ---> Applying smoothing ... DONE!')
            # -------------------------------------------------------------------------------------

            # -------------------------------------------------------------------------------------
            # final masking
            map_t_da.values = map_t_da.values * da_domain_target.values
            # -------------------------------------------------------------------------------------

            # -------------------------------------------------------------------------------------
            # Write to file
            path_geotiff_output = os.path.join(data_settings['data']['outcome']['folder'],
                                               data_settings['data']['outcome']['filename'])
            tag_filled = {'outcome_sub_path_time': time_date,
                          'outcome_datetime': time_date}
            path_geotiff_output = \
                fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
            if os.path.isdir(os.path.split(path_geotiff_output)[0]) is False:
                os.makedirs(os.path.split(path_geotiff_output)[0])
            write_file_tiff(path_geotiff_output, map_t_da.values, wide_domain_target, high_domain_target,
                            transform_domain_target, 'EPSG:4326')
            logging.info(' --> Map saved for time: ' + time_date.strftime("%Y/%m/%d %H:%M") + ' at: ' + path_geotiff_output)
            # -------------------------------------------------------------------------------------

        else:
            logging.warning(' ---> Map not created for timestep ' + time_date.strftime("%Y/%m/%d %H:%M") + ' due to missing data!')


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