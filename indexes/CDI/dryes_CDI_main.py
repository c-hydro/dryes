"""
DRYES Drought Metrics Tool - CDI Combined Drought Index following https://doi.org/10.1016/j.ejrh.2023.101404
__date__ = '20231002'
__version__ = '1.0.1'
__author__ =
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org),
        'Fabio Delogu' (fabio.delogu@cimafoundation.org),
        'Michel Isabellon (michel.isabellon@cimafoundation.org),
        'Edoardo Cremonese' (edoardo.cremonese@cimafoundation.org)

__library__ = 'dryes'

General command line:
python dryes_CDI_main.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM"

Version(s):
20231002 (1.0.1) --> Added mkdir in output folder
20230927 (1.0.0) --> First release
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


from dryes_CDI_utils_json import read_file_json
from dryes_CDI_geo import read_file_raster
from dryes_CDI_utils_generic import fill_tags2string
from dryes_CDI_tiff import write_file_tiff

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'CDI DROUGHT METRIC'
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
    [file_script, file_settings, time_arg_now] = get_args()

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
    time_arg_now_pd = pd.to_datetime(time_arg_now)

    # -------------------------------------------------------------------------------------
    # Load var1 (and resample if needed)
    path_data_var1 = os.path.join(data_settings['data']['input']['var_1']['folder'],
                             data_settings['data']['input']['var_1']['filename'])
    tag_filled = {'source_gridded_var1_sub_path_time': time_arg_now_pd,
                  'source_gridded_var1_datetime': time_arg_now_pd}
    path_data_var1 = fill_tags2string(path_data_var1, data_settings['algorithm']['template'], tag_filled)

    if os.path.isfile(path_data_var1):

        data_var1 = rioxarray.open_rasterio(path_data_var1)
        data_var1 = np.squeeze(data_var1)

        logging.info(' --> Loaded var1 from ' + path_data_var1)

        if data_var1.shape != da_domain_in.shape:
            coordinates_target = {
                data_var1.dims[0]: da_domain_in[da_domain_in.dims[0]].values,
                data_var1.dims[1]: da_domain_in[da_domain_in.dims[1]].values}
            data_this_day_var1 = data_var1.interp(coordinates_target, method='nearest')
            logging.info(' --> Resampled var1')
        else:
            data_this_day_var1 = data_var1

        data_var1_values = data_this_day_var1.values

    else:
        logging.error(' ===> Var1 not found at' + path_data_var1)
        raise ValueError(' ===> Var1 not found!')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load var2 (and resample if needed)
    path_data_var2 = os.path.join(data_settings['data']['input']['var_2']['folder'],
                                  data_settings['data']['input']['var_2']['filename'])
    tag_filled = {'source_gridded_var2_sub_path_time': time_arg_now_pd,
                  'source_gridded_var2_datetime': time_arg_now_pd}
    path_data_var2 = fill_tags2string(path_data_var2, data_settings['algorithm']['template'], tag_filled)

    if os.path.isfile(path_data_var2):

        data_var2 = rioxarray.open_rasterio(path_data_var2)
        data_var2 = np.squeeze(data_var2)

        logging.info(' --> Loaded var2 from ' + path_data_var2)

        if data_var2.shape != da_domain_in.shape:
            coordinates_target = {
                data_var2.dims[0]: da_domain_in[da_domain_in.dims[0]].values,
                data_var2.dims[1]: da_domain_in[da_domain_in.dims[1]].values}
            data_this_day_var2 = data_var2.interp(coordinates_target, method='nearest')
            logging.info(' --> Resampled var2')
        else:
            data_this_day_var2 = data_var2

        data_var2_values = data_this_day_var2.values

    else:
        logging.error(' ===> Var1 not found at' + path_data_var1)
        raise ValueError(' ===> Var1 not found!')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Compute CDI with thresholds
    thresholds_var1 = data_settings['index_info']['threshold_var1']
    thresholds_var2 = data_settings['index_info']['threshold_var2']
    scores = data_settings['index_info']['scores']
    CDI = np.zeros(shape=(da_domain_in.shape[0], da_domain_in.shape[1])) * np.nan
    for i_row, value_scores in enumerate(scores):
        for i_column, values_this_item in enumerate(value_scores):

            CDI[(data_var1_values > thresholds_var1[i_row]) &
                (data_var1_values <= thresholds_var1[i_row + 1]) &
                (data_var2_values <= thresholds_var2[i_column]) &
                (data_var2_values > thresholds_var2[i_column + 1])] = values_this_item

            #print('Row: ' + str(thresholds_var1[i_row]) + ' to ' + str(thresholds_var1[i_row + 1]))
            #print('Column: ' + str(thresholds_var2[i_column]) + ' to ' + (str(thresholds_var2[i_column + 1])))
            #plt.figure()
            #plt.imshow(CDI)
            #plt.savefig('CDI.png')
            #plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Mask
    if data_settings['algorithm']['flags']['mask_results_static']:
        CDI[da_domain_in.values == 0] = np.nan
        CDI[np.isnan(da_domain_in.values)] = np.nan
        logging.info(' --> CDI masked')

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # export to tiff
    path_geotiff_output = data_settings['data']['outcome']['path_output_results']
    tag_filled = {'outcome_datetime': pd.to_datetime(time_arg_now),
                  'outcome_sub_path_time': pd.to_datetime(time_arg_now)}
    path_geotiff_output = \
                fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
    dir, filename = os.path.split(path_geotiff_output)
    if os.path.isdir(dir) is False:
        os.makedirs(dir)
    write_file_tiff(path_geotiff_output, CDI, wide_domain_in, high_domain_in, transform_domain_in, 'EPSG:4326')
    logging.info(' --> CDI saved at ' + path_geotiff_output)
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
