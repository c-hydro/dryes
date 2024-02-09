"""
DRYES Drought Metrics Tool - Tool to compute daily PET according to Hargreaves-Samani equation
__date__ = '20240208'
__version__ = '1.0.0'
__author__ =
        'Francesco Avanzi (francesco.avanzi@cimafoundation.org'),
        'Matilde Torrassa (matilde.torrassa@cimafoundation.org)'

__library__ = 'dryes'

General command line:
python dryes_tool_pet_HS.py -settings_file "dryes_tool_pet_HS.json" -time_now "yyyy-mm-dd 00:00"

Version(s):
20240208 (1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import logging
from os.path import join
from argparse import ArgumentParser
import numpy as np
import os
import rasterio
from time import time, strftime, gmtime
import matplotlib as mpl
import matplotlib.pylab as plt

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from dryes_tool_tool_pet_HS_json import read_file_json
from dryes_tool_tool_pet_HS_time import set_time
from dryes_tool_tool_pet_HS_geo import read_file_raster
from dryes_tool_pet_HS_generic import fill_tags2string
from dryes_tool_tool_pet_HS_tiff import write_file_tiff

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'PET HS'
alg_version = '1.0.0'
alg_release = '2024-02-08'
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
    # Load grid
    logging.info(' --> Load target grid ... ')
    da_domain_target, wide_domain_target, high_domain_target, proj_domain_target, transform_domain_target, \
        bounding_box_domain_target, no_data_domain_target, crs_domain_target, lons_target, lats_target = \
        read_file_raster(data_settings['data']['input']['input_grid'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load target grid ... DONE')

    # convert lat to radiant
    lat = lats_target * np.pi / 180

    # plt.figure()
    # plt.imshow(da_domain_target.values)
    # plt.savefig('da_domain_in.png')
    # plt.close()

    # -------------------------------------------------------------------------------------
    # Iterate over time steps
    for time_i, time_date in enumerate(time_range):

        file_not_found = 0

        #open T air max
        path_data = os.path.join(data_settings['data']['input']['folder'],
        data_settings['data']['input']['filename_max'])
        tag_filled = {'source_gridded_sub_path_time': time_date,
                      'source_gridded_datetime': time_date}
        path_data = fill_tags2string(path_data, data_settings['algorithm']['template'], tag_filled)
        if os.path.isfile(path_data):
            with rasterio.open(path_data) as src:
                Tmax = src.read(1)
        else:
            logging.warning('File' + path_data + ' NOT FOUND')
            file_not_found = 1

        # open T air min
        path_data = os.path.join(data_settings['data']['input']['folder'],
                                     data_settings['data']['input']['filename_min'])
        tag_filled = {'source_gridded_sub_path_time': time_date,
                          'source_gridded_datetime': time_date}
        path_data = fill_tags2string(path_data, data_settings['algorithm']['template'], tag_filled)
        if os.path.isfile(path_data):
            with rasterio.open(path_data) as src:
                Tmin = src.read(1)
        else:
            logging.warning('File' + path_data + ' NOT FOUND')
            file_not_found = 1

        # open T air avg
        path_data = os.path.join(data_settings['data']['input']['folder'],
                                     data_settings['data']['input']['filename_mean'])
        tag_filled = {'source_gridded_sub_path_time': time_date,
                          'source_gridded_datetime': time_date}
        path_data = fill_tags2string(path_data, data_settings['algorithm']['template'], tag_filled)
        if os.path.isfile(path_data):
            with rasterio.open(path_data) as src:
                Tavg = src.read(1)
        else:
            logging.warning('File' + path_data + ' NOT FOUND')
            file_not_found = 1

        if file_not_found == 0:

            # Compute daily Extraterrestrial Radiation
            DOY = time_date.timetuple().tm_yday
            sol_dec = 0.409 * (np.sin((2 * np.pi * DOY / 365) - 1.39))
            ws = np.arccos(-np.tan(lat) * np.tan(sol_dec))
            dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * DOY)
            Ra = 0.408 * 24 * 60 / np.pi * 0.0820 * dr * (
                            (ws * np.sin(lat) * np.sin(sol_dec)) + (np.cos(lat) * np.cos(sol_dec) * np.sin(ws)))

            # compute daily PET with standard H-S
            pet = 0.0023 * Ra * ((Tmax - Tmin) ** 0.5) * (Tavg + 17.8)
            pet[pet < 0] = 0
            logging.info(f' --> daily PET computed for date {time_date.strftime("%Y/%m/%d")}')

            # masking
            if data_settings['data']['outcome']['mask']:
                pet = pet * da_domain_target.values

            # Write to file
            path_geotiff_output = os.path.join(data_settings['data']['outcome']['path'])
            tag_filled = {'outcome_sub_path_time': time_date,
                          'outcome_datetime': time_date}
            path_geotiff_output = \
                fill_tags2string(path_geotiff_output, data_settings['algorithm']['template'], tag_filled)
            if os.path.isdir(os.path.split(path_geotiff_output)[0]) is False:
                os.makedirs(os.path.split(path_geotiff_output)[0])
            write_file_tiff(path_geotiff_output, pet, wide_domain_target, high_domain_target,
                            transform_domain_target, 'EPSG:4326')
            logging.info(' --> Map saved for time: ' + time_date.strftime("%Y/%m/%d %H:%M") + ' at: ' + path_geotiff_output)

        else:
            logging.warning('PET for day ' + time_date.strftime("%Y/%m/%d %H:%M") + ' not computed due to missing input files')

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
