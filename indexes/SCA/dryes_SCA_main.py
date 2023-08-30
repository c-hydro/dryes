"""
DRYES Drought Metrics Tool - Snow-Covered Area Anomaly by mountain regions
__date__ = '20230830'
__version__ = '1.0.1'
__author__ =
        'Francesco Avanzi' (francesco.avanzi@cimafoundation.org',
        'Fabio Delogu' (fabio.delogu@cimafoundation.org',
        'Michel Isabellon (michel.isabellon@cimafoundation.org'

__library__ = 'dryes'

General command line:
python dryes_SCA_main.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM" -time_history_start "yyyy-mm-dd HH:MM" -time_history_end  "yyyy-mm-dd HH:MM"

Version(s):
20230830 (1.0.1) --> Corrected minor bug in the management of leap yrs
20230629 (1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Library
import logging
from os.path import join
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from time import time, strftime, gmtime
import datetime
import xarray as xr

from dryes_SCA_utils_json import read_file_json
from dryes_SCA_geo import read_file_raster, avg_SCA_by_mountain_region
from dryes_SCA_utils_time import enforce_midnight
from dryes_SCA_utils_generic import fill_tags2string

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'DRYES'
alg_name = 'SCA DROUGHT METRIC'
alg_version = '1.0.1'
alg_release = '2023-08-30'
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
    logging.info('[' + alg_project + '] Historical Period: ' + time_arg_history_start + ' to ' + time_arg_history_end + ' GMT')
    logging.info('[' + alg_project + '] Start Program ... ')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load list of mountain groups & mountain mask
    logging.info(' --> Load target grid ... ')
    da_domain_in, wide_domain_in, high_domain_in, proj_domain_in, transform_domain_in, \
        bounding_box_domain_in, no_data_domain_in, crs_domain_in =\
        read_file_raster(data_settings['data']['input']['mountain_mask'])
    logging.info(' --> Load target grid ... DONE')

    # plt.figure()
    # plt.imshow(da_domain_in.values)
    # plt.savefig('da_domain_in.png')
    # plt.close()

    list_mountains = pd.read_csv(data_settings['data']['input']['table_mountain_groups'])
    list_mountains = list_mountains.reset_index()  # make sure indexes pair with number of rows
    # -------------------------------------------------------------------------------------
    # Time algorithm information
    start_time = time()

    # we check times and enforce them to midnight if needed
    if pd.to_datetime(time_arg_now).hour is not 0:
        time_arg_now = enforce_midnight(time_arg_now)
        logging.warning(' ==> time_arg_now enforced to midnight!')
    if pd.to_datetime(time_arg_history_start).hour is not 0:
        time_arg_history_start = enforce_midnight(time_arg_history_start)
        logging.warning(' ==> time_arg_history_start enforced to midnight!')
    if pd.to_datetime(time_arg_history_end).hour is not 0:
        time_arg_history_end = enforce_midnight(time_arg_history_end)
        logging.warning(' ==> time_arg_history_end enforced to midnight!')

    # we build the daily historical datetimeindex object
    period_history_daily = pd.date_range(time_arg_history_start, time_arg_history_end, freq="D")  # for loading data

    # we build the daily now datetimeindex object from the onset of the water year
    time_arg_now_pd = pd.to_datetime(time_arg_now)
    if time_arg_now_pd.month >= data_settings['index_info']['month_start_WY']:
        period_this_wy_daily = \
            pd.date_range(datetime.date(time_arg_now_pd.year, data_settings['index_info']['month_start_WY'],
                                        1), time_arg_now_pd, freq="D")
    else:
        period_this_wy_daily = \
            pd.date_range(datetime.date(time_arg_now_pd.year - 1, data_settings['index_info']['month_start_WY'],
                                        1), time_arg_now_pd, freq="D")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load SCA data from h10, resample and compute avg values across mountain regions
    # history
    SCA_by_regions_history = avg_SCA_by_mountain_region(period_history_daily, data_settings,
                                                        list_mountains, da_domain_in,
                                                        crs_domain_in, transform_domain_in)
    logging.info(' --> Load historical data ... DONE')

    # real time
    SCA_by_regions_this_wy = avg_SCA_by_mountain_region(period_this_wy_daily, data_settings,
                                                        list_mountains, da_domain_in,
                                                        crs_domain_in, transform_domain_in)
    logging.info(' --> Load current-year data ... DONE')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Compute historical percentiles and plot
    SCA_by_regions_history['Year'] = SCA_by_regions_history.index.year
    SCA_by_regions_history['Month'] = SCA_by_regions_history.index.month
    SCA_by_regions_history['Day'] = SCA_by_regions_history.index.day

    for (columnName, columnData) in SCA_by_regions_history.iteritems():

        if list_mountains.name_mm.str.contains(columnName).any():
            # we check if columnName is one of the mountain groups for which we want to compute statistics

            if (time_arg_now_pd.month == 2) &  (time_arg_now_pd.day == 29) is False:
                # if it's 29/2, we skip computations (we do not support 29/2 during leap yrs)

                logging.info(' --> Working on ... ' + columnName)

                SCA_by_regions_history_stat = pd.DataFrame()

                # We compute statistics on the historical period
                SCA_by_regions_history_stat['Min'] = \
                    SCA_by_regions_history.groupby(['Month', 'Day'])[columnName].min()
                SCA_by_regions_history_stat['Q25'] = \
                                            SCA_by_regions_history.groupby(['Month', 'Day'])[columnName].apply(lambda x: np.nanpercentile(x.astype(float), 25))
                SCA_by_regions_history_stat['Q50'] = \
                    SCA_by_regions_history.groupby(['Month', 'Day'])[columnName].apply(lambda x: np.nanpercentile(x.astype(float), 50))
                SCA_by_regions_history_stat['Q75'] = \
                    SCA_by_regions_history.groupby(['Month', 'Day'])[columnName].apply(lambda x: np.nanpercentile(x.astype(float), 75))
                SCA_by_regions_history_stat['Max'] = \
                    SCA_by_regions_history.groupby(['Month', 'Day'])[columnName].max()
                SCA_by_regions_history_stat['Std'] = \
                    SCA_by_regions_history.groupby(['Month', 'Day'])[columnName].apply(lambda x: np.std(x.astype(float)))

                # We remove Feb 29 from the historical pool
                if SCA_by_regions_history_stat.index.isin([(2,29)]).any():
                    SCA_by_regions_history_stat =  SCA_by_regions_history_stat.drop(index=(2, 29))

                # We re-organize the datarange in Sep-Aug index using a fake, non-leap datarange
                SCA_by_regions_history_stat.index = pd.to_datetime(SCA_by_regions_history_stat.index, format="(%m, %d)") #we create new index with datetimes
                SCA_by_regions_history_stat.index = \
                    SCA_by_regions_history_stat.index.map(lambda x: x.replace(year=2021) if (x.month >= 9) else x.replace(year=2022))
                SCA_by_regions_history_stat = SCA_by_regions_history_stat.sort_index()

                # Now we collect data for the current year, also assign the same fake datetime array, and then merge df
                SCA_by_regions_this_wy_and_region = SCA_by_regions_this_wy[columnName]
                SCA_by_regions_this_wy_and_region.index = \
                    SCA_by_regions_this_wy_and_region.index.map(
                        lambda x: x.replace(year=2021) if (x.month >= 9) else x.replace(year=2022))
                SCA_merged = pd.merge(SCA_by_regions_history_stat, SCA_by_regions_this_wy_and_region,
                                      how='left', left_index=True, right_index=True)

                # We compute mov mean
                SCA_merged_rolling = SCA_merged.rolling(data_settings['index_info']['days_moving_mean_SCA'],
                                                        min_periods=data_settings['index_info']['min_data_for_mocing_mean'],
                                                        center=True,closed='both').mean()

                # We compute anomaly for this day
                SCA_now = SCA_merged_rolling.loc[(SCA_merged_rolling.index.month == time_arg_now_pd.month) &
                                                 (SCA_merged_rolling.index.day == time_arg_now_pd.day),columnName]
                SCA_Q50_this_day = SCA_merged_rolling.loc[(SCA_merged_rolling.index.month == time_arg_now_pd.month) &
                                                 (SCA_merged_rolling.index.day == time_arg_now_pd.day),'Q50']
                SCA_std_this_day = SCA_merged_rolling.loc[(SCA_merged_rolling.index.month == time_arg_now_pd.month) &
                                                 (SCA_merged_rolling.index.day == time_arg_now_pd.day),'Std']
                Anomaly = (SCA_now[0] - SCA_Q50_this_day[0])/SCA_Q50_this_day[0]*100
                Zscore = (SCA_now[0] - SCA_Q50_this_day[0]) / SCA_std_this_day[0]

                logging.info(' --> Anomaly for ... ' + columnName + ': ' + str(np.round(Anomaly,2)) + '%')
                logging.info(' --> Z score for ... ' + columnName + ': ' + str(np.round(Zscore, 2)))

                # We generate plot
                if pd.to_datetime(time_arg_now).month >= 9:
                    str_label_WYnow = str(pd.to_datetime(time_arg_now).year) + ' ' + str(pd.to_datetime(time_arg_now).year + 1)
                else:
                     str_label_WYnow = str(pd.to_datetime(time_arg_now).year - 1) + ' ' + str(pd.to_datetime(time_arg_now).year)

                plt.figure(figsize=(30, 15))
                plt.fill_between(SCA_merged_rolling.index, SCA_merged_rolling['Q25'], SCA_merged_rolling['Q75'],
                                 alpha=0.5, color='gray',label='First-Third historical quartile')
                plt.plot(SCA_merged_rolling.index, SCA_merged_rolling['Q50'], color='black',linestyle='-',linewidth=2
                         ,label='Historical median')
                plt.plot(SCA_merged_rolling.index, SCA_merged_rolling[columnName] ,color='red', linewidth=3, marker='o', label=str_label_WYnow)
                handles, labels = plt.gca().get_legend_handles_labels()
                order = [2, 0, 1]
                plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=25) # we make sure that the order of legend items is correct

                plt.grid()
                plt.ylabel('Snow-covered area, -', fontsize=30)
                plt.tick_params(labelsize=30)
                plt.title(columnName + ', ' + time_arg_now + ' - SE-E-SEVIRI (H10) anomaly: ' + str(np.round(Anomaly, 2)) +
                         '%, Z-score: ' + str(np.round(Zscore, 2)), fontsize=30)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b'))

                path_fig = data_settings['data']['outcome']['path_output_results']
                tag_filled = {'domain': columnName, 'outcome_datetime': time_arg_now_pd, 'outcome_sub_path_time': time_arg_now_pd}
                path_fig = fill_tags2string(path_fig, data_settings['algorithm']['template'], tag_filled)
                dir, filename = os.path.split(path_fig)
                if os.path.isdir(dir) is False:
                    os.makedirs(dir)
                plt.savefig(path_fig)
                plt.close()
                logging.info(' --> Figure saved at: ' + path_fig)

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
