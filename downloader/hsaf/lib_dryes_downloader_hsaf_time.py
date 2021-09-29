"""
Library Features:

Name:          lib_dryes_downloader_time
Author(s):     Francesco Avanzi (francesco.avanzi@cimafoundation.org), Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210929'
Version:       '1.0.0'
"""
#################################################################################
# Library
import os
import pandas as pd
import logging
#################################################################################



# -------------------------------------------------------------------------------------
# Method to define data time range
def set_data_time(time_step, time_settings):
    time_period_obs = time_settings['time_observed_period']
    time_period_for = time_settings['time_forecast_period']
    time_freq_obs = time_settings['time_observed_frequency']
    time_freq_for = time_settings['time_forecast_frequency']
    time_format = time_settings['time_format']

    time_step_obs = time_step

    if time_step.hour > 0:
        time_step_to = time_step_obs
        time_step_from = time_step.floor('D')
        time_range_obs = pd.date_range(start=time_step_from, end=time_step_to, freq=time_freq_obs)
    else:
        time_range_obs = pd.date_range(end=time_step_obs, periods=time_period_obs, freq=time_freq_obs)

    time_step_for = pd.date_range([time_step][0], periods=2, freq=time_freq_for)[-1]
    time_range_for = pd.date_range(start=time_step_for, periods=time_period_for, freq=time_freq_for)

    time_range_data = time_range_obs.union(time_range_for)

    time_range_start = time_range_data[0].strftime(time_format)
    time_range_end = time_range_data[-1].strftime(time_format)

    return time_range_data, time_range_start, time_range_end

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to define run time range
def set_run_time(time_alg, time_settings, time_ascending=False):
    time_set = time_settings['time_now']
    time_freq = time_settings['time_frequency']
    time_round = time_settings['time_rounding']
    time_period = time_settings['time_period']

    if time_alg and time_set:
        time_now = time_alg
    elif time_alg is None and time_set:
        time_now = time_set
    elif time_alg and time_set is None:
        time_now = time_alg
    else:
        logging.error(' ===> TimeNow is not correctly set!')
        raise IOError('TimeNow is undefined! Check your settings or algorithm args!')

    time_now_raw = pd.Timestamp(time_now)
    time_now_round = time_now_raw.floor(time_round)
    time_day_round = time_now_raw.floor('D')

    if time_now_round > time_day_round:
        time_last = pd.DatetimeIndex([time_now_round])
    else:
        time_last = None

    if time_period > 0:
        time_range = pd.date_range(end=time_day_round, periods=time_period, freq=time_freq)
    else:
        logging.warning(' ===> TimePeriod must be greater then 0. TimePeriod is set automatically to 1')
        time_range = pd.DatetimeIndex([time_day_round], freq=time_freq)

    if time_last is not None:
        time_period = time_range.union(time_last)
    else:
        time_period = time_range

    time_period = time_period.sort_values(ascending=time_ascending)

    return time_now_round, time_period
# -------------------------------------------------------------------------------------






