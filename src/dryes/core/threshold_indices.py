import xarray as xr
import subprocess
import numpy as np
import datetime as dt
import os

from typing import Generator

def calc_dintensity(data: np.ndarray, threshold: np.ndarray, direction: int = 1, get_hits: bool = True) -> dict[str:np.ndarray]:
    """
    Calculate the daily intensity of the data above a given threshold.
    The function returns the daily intensity and a boolean array indicating whether the data is above the threshold.
    
    Data and threshold can contain multiple variables (e.g. Tmax and Tmin) in the first dimension.
    Daily intensity for each variable is calculated as max(0, data - threshold * direction).
    If multiple variables are present, the daily intensity returned is the average of the daily intensity of all variables.

    The hit array is a boolean array indicating whether the data is above/below the threshold.
    The hit array is 1 if the data is above the threshold (if direction = 1) or below the threshold (if direction = -1), and 0 otherwise.
    If multiple variables are present, the hit array is 1 only if all variables are above/below the threshold.

    Parameters:
        data (np.ndarray): The input data array. If the data is at least 3D, the first dimension represents different variables (e.g. Tmax and Tmin).
        threshold (np.ndarray): The threshold array. Must have the same shape as the data array.
        direction (int): The direction of the threshold. 1 for above, -1 for below. Default is 1.
        get_hits (bool): If True, return the hits as well. Default is True.

    Returns:
        (dict): A dictionary containing the daily intensity and hit arrays. The keys are:
            - 'dintensity' (np.ndarray): The daily intensity array.
            - 'ishit'      (np.ndarray): The hit array (only if get_hits is True).
    
    Raises:
        ValueError: If the data and threshold arrays do not have the same shape.
    """
    
    # ensure the data and threshold have the same shape
    if data.shape != threshold.shape:
        raise ValueError("The data and threshold arrays must have the same shape.")

    # calculate the intensity
    intensity = np.maximum(0, data - threshold * direction)

    # get the hits
    if get_hits:
        hits = (intensity > 0).astype(int)
        # if the data is 3D, only consider hits where all variables are above the threshold
        if len(hits.shape) >= 3:
            hits = np.min(hits, axis=0)

    if len(intensity.shape) >= 3:
        # if the data is 3D, calculate the average over the first dimension
        intensity = np.mean(intensity, axis=0)
    
    if get_hits:
        # return the intensity and hits
        return {'dintensity': intensity, 'ishit': hits}
    else:
        # return only the intensity
        return {'dintensity': intensity}

def pool_index(daily_index: dict[str:np.ndarray], 
               current_pooled_index: dict[str:np.ndarray],
               n: int = 1,
               min_duration: int = 3,
               min_interval: int = 1,
               look_ahead: bool = True,
               count_with_pools: bool = True,
               pool_if_uncertain: int = 1) -> dict[str:np.ndarray]:
    """
    Pools the daily index to create a pooled index.
    The pooling rules are consistent with Lavayasse et al. (2017).
    To better understand the pooling procedure, it is useful to distinguish between:
        - 'hit' day  : a day that is above the threshold
        - 'spell'    : a sequence of hit days separated by at most "min_interval" pool days
        - 'pool' day : a day that is not a hit day, but is part of a spell, there can by a maximum of "min_interval" pool days between two hit days
        - 'event'    : a spell that is longer than the "min_duration"

    Parameters:
        daily_index (dict): A dictionary containing the daily index data. The expected keys are:
            - 'dintensity' (np.ndarray): A 3D array of daily intensity values.
            - 'ishit'      (np.ndarray): A 3D array indicating whether a day is a hit day (1) or not (0).
            in both cases, the first dimension is the time dimension, the second and third dimensions are the lat/lon dimensions.
            both layers have the same shape and contain: n timesteps that refer to the period of interest, 
            plus additional timesteps into the future in case look_ahead is True,
            the number of future timesteps should be at least (min_duration-1)*(min_interval+1) days.
        current_pooled_index (dict): A dictionary containing the current pooled index data. The expected keys are (see returns for more details):
            - 'intensity'  (np.ndarray): A 2D array containing the intensity of ongoing events.
            - 'duration'   (np.ndarray): A 2D array containing the duration of ongoing events.
            - 'interval'   (np.ndarray): A 2D array containing the interval (i.e. number of days) since the last hit day.
            - 'nhits'      (np.ndarray): A 2D array containing the number of hit days of ongoing spells.
            - 'sintensity' (np.ndarray): A 2D array containing the intensity of ongoing spells.
            - 'sduration'  (np.ndarray): A 2D array containing the duration of ongoing spells.
        n (int): The number of timesteps to pool over before returning the output. Default is 1 (i.e. output every day).
        min_duration (int): The minimum duration of a spell to be considered an event. Default is 3.
        min_interval (int): The maximum number of pool days between two hit days to be considered a spell. Default is 1.
        look_ahead (bool): If True, look ahead in the time series to determine if a spell will be an event in the future. Default is True.
        count_with_pools (bool): If True, add the intensity of pool days to the event intensity and increase the duration of the event for pool days too. Default is True.
        pool_if_uncertain (int): Determines how to behave in the case of uncertain pool days (usually if look_ahead=False, but also if there is no future data available)
            If 0, pool days that are uncertain (i.e. we don't know if the spell will continue or not) are not considered pool days.
            If 1, pool days that are uncertain are considered as pool days, but duration and intensity are not increased, even if count_with_pools=True.
            If 2, pool days that are uncertain are considered as pool days.

    Returns:
        (dict): A dictionary containing the pooled index data after n timesteps. The expected keys are:
            - 'intensity'  (np.ndarray): A 2D array containing the intensity of ongoing events.
            - 'duration'   (np.ndarray): A 2D array containing the duration of ongoing events.
            - 'interval'   (np.ndarray): A 2D array containing the interval (i.e. number of days) since the last hit day.
            - 'nhits'      (np.ndarray): A 2D array containing the number of hit days of ongoing spells.
            - 'sintensity' (np.ndarray): A 2D array containing the intensity of ongoing spells.
            - 'sduration'  (np.ndarray): A 2D array containing the duration of ongoing spells.
            - 'case'       (np.ndarray): A 2D array containing a debugging variable. The values are:
                 0 : ERROR! THIS SHOULD NOT HAPPEN!
                 1 : not a hit day that is not part (or the end of) a spell
                 2 : pool day that is part of an event: min_duration already reached (only if count_with_pools is True)
                 3 : pool day that will be part of an event: min_duration will be reached in the future (only if count_with_pools is True and look_ahead is True)
                 4 : pool day that is not part of an event: min_duration is not reached and will not be reached in the future (if count_with_pools is False, this is any pool day)
                 5 : not hot day that could be a pool day (we don't know yet)
                 6 : not hot day that is the end of a spell (reset the counters)
                12 : hot day that is part of an event: min_duration already reached
                13 : hot day that will be part of an event: min_duration will be reached in the future (only if look_ahead is True)
                14 : hot day that is not part of an event: min_duration is not reached and will not be reached in the future
                15 : hot day that is not yet part of an event: min_duration is not reached, but we don't know yet if it will be reached in the future
    
    Raise:
        ValueError: If the daily_index components do not have the same shape or if they have less than n timesteps.
                
    References:
        Lavaysse, C., Camalleri, C., Dosio, A., van der Schrier, G., Toreti, A., & Vogt, J. (2017).
        Towards a monitoring system of temperature extremes in Europe. Natural Hazards and Earth System Sciences Discussions, 2017, 1-29.
    """

    # extract the current daily index (intensity and hit)
    daily_intensity = daily_index['dintensity']
    daily_hit       = daily_index['ishit']

    # ensure the daily_index components have the same shape and at least n timesteps
    if daily_intensity.shape != daily_hit.shape:
        raise ValueError("The daily_index components must have the same shape.")
    if daily_intensity.shape[0] < n:
        raise ValueError("The daily_index components must have at least n timesteps.")

    # if look_ahead is false, we only need the first n days of the daily index
    if not look_ahead:
        daily_intensity = daily_intensity[:n]
        daily_hit       = daily_hit[:n]
    # if look_ahead is true, we need the first n days + (min_duration-1)*(min_interval+1),
    # if we don't have enough data, we use what we have
    else:
        look_ahead_length = (min_duration-1) * (min_interval+1)
        if daily_intensity.shape[0] >= n + look_ahead_length:
            daily_intensity = daily_intensity[:n+look_ahead_length]
            daily_hit       = daily_hit[:n+look_ahead_length]

    # get the current pooled index components (or initialise them if they are None)
    if (intensity := current_pooled_index.get('intensity')) is None:
        intensity = np.zeros_like(daily_intensity[0])

    if (duration := current_pooled_index.get('duration')) is None:
        duration = np.zeros_like(daily_intensity[0], dtype=int)
    else:
        duration = duration.astype(int)

    if (interval := current_pooled_index.get('interval')) is None:
        interval = np.zeros_like(daily_intensity[0], dtype=int) + min_interval + 1
    else:
        interval = interval.astype(int)

    if (nhits := current_pooled_index.get('nhits')) is None:
        nhits = np.zeros_like(daily_intensity[0], dtype=int)
    else:
        nhits = nhits.astype(int)

    if (sintensity := current_pooled_index.get('sintensity')) is None:
        sintensity = np.zeros_like(daily_intensity[0])

    if (sduration := current_pooled_index.get('sduration')) is None:
        sduration = np.zeros_like(daily_intensity[0], dtype=int)
    else:
        sduration = sduration.astype(int)

    # initialise the case variable as a 2D array of zeros
    this_case = np.zeros_like(daily_intensity[0], dtype=int)

    # loop throught the n days we need to pool before returning
    for i in range(n):
        # first check where we have a hit day or not
        is_hit = daily_hit[i] == 1

        # where there is a hit day, set the interval since the last hit day to 0 and increase the number of hit days in this spell by 1
        # this is valid for all cases > 10
        interval[is_hit]  = 0
        nhits[is_hit]    += 1

        # everywhere else, increase the interval
        # this is valid for all cases < 10
        interval[np.logical_not(is_hit)] += 1

        # where we have the end of a spell, we need to check if this is a pool day or not
        is_end_of_spell = np.logical_and(np.logical_not(is_hit), nhits > 0)

        # case 01: not hit day that is not part of or the end of a spell -> do nothing!
        c01 = np.logical_and(np.logical_not(is_hit), np.logical_not(is_end_of_spell))
        this_case[c01] = 1

        # if we look ahead, let's check when the next spell begins (to determine if end_of_spell days are pool days)
        # future_interval is the number of days until the next hit day (if there is one)
        # future_interval_iscertain is True if we are certain that the next hit day 
        #   (i.e. there is at least one hit day in the future, otherwise future_interval > length of future we have)
        future_interval = np.zeros_like(daily_hit[i])
        future_interval_iscertain = np.zeros_like(daily_hit[i])
        if look_ahead and i < daily_hit.shape[0] - 1:
            future_interval[is_end_of_spell] = np.argmax(daily_hit[i+1:,is_end_of_spell], axis = 0)
            future_interval_iscertain[is_end_of_spell] = daily_hit[i+1:,is_end_of_spell].any(axis = 0)
            future_interval[np.logical_and(is_end_of_spell,np.logical_not(future_interval_iscertain))] = daily_hit.shape[0] - i - 1

        # pool days are where the interval (past + eventually, future) is less than the min_interval
        is_pool_day    = np.logical_and(is_end_of_spell, interval + future_interval <= min_interval)
        maybe_pool_day = np.logical_and.reduce([is_pool_day, np.logical_not(future_interval_iscertain)])

        # unless pool_if_uncertain is 2, we set the maybe_pool_day to not pool days, because we need to treat them differently
        if pool_if_uncertain != 2: is_pool_day[maybe_pool_day] = False
        
        # for all the points that are in a spell (either hit or pool days), we need to check if this spell is an event
        # (to determine if we increase duration and interval or not)
        # a spell is an event if the number of hit days *will* be longer than min_duration at any point in the future
        # future_nhits is the number of hit days in the future (if there is one)
        # future_nhits_iscertain is True if we are certain that the next hit day
        #   (i.e. if the spell ends in the future we know, otherwise future_nhits is a lower bound)
        future_nhits = np.zeros_like(nhits)
        future_nhits_iscertain = np.zeros_like(nhits)
        if look_ahead and i < daily_hit.shape[0]-1:
            
            # get the points that we need to look at
            # thats all the points that are in a spell and that have not yet reached min_duration
            # if count_with_pools is True, we also need to look at the pool days that are in a spell
            if count_with_pools:
                xx,yy = np.where(np.logical_and(np.logical_or(is_hit, is_pool_day), nhits < min_duration))
            # if count_with_pools is False, we only need to look at the hit days, because we don't increase duration and intensity for pool days
            else:
                xx,yy = np.where(np.logical_and(is_hit, nhits < min_duration))

            # loop through each point
            for x, y in zip(xx, yy):
                this_future = daily_hit[i+1:, x, y]
                # and loop through all future timesteps to find the end of the current spell (if it is there),
                # that is a (min_interval+1) long set of zeros
                for j in range(len(this_future) - min_interval):
                    if np.all(this_future[j:j+(min_interval+1)] == 0):
                        if j > 0: future_nhits[x, y] = np.cumsum(this_future[:j])[-1]
                        future_nhits_iscertain[x, y] = 1
                        break
                else:
                    # if we don't find the end of the spell, it means it is still going on through the end of the future data
                    # so we take the whole future data
                    future_nhits[x, y] = np.cumsum(this_future)[-1]

        # case 12 : hot day with nhits >= min_duration (in the past)
        c12 = np.logical_and(is_hit, nhits >= min_duration)
        
        # case 13 : hot day with nhits + future_nhits >= min_duration (in the future)
        c13 = np.logical_and(is_hit, nhits + future_nhits >= min_duration)
        this_case[c13] = 13
        this_case[c12] = 12 # 12 is a subset of 13 and they have the same outcome, it is still helpful to keep track of it

        # case 14: hot day with nhits + future_nhits < min_duration (not an event)
        c14 = np.logical_and(is_hit, nhits + future_nhits < min_duration)
        this_case[c14] = 14
        
        # case 15: hot day that we don't not yet if it is part of an event (subset of 14, where we are uncertain)
        c15 = np.logical_and(c14, np.logical_not(future_nhits_iscertain))
        this_case[c15] = 15
        
        # case 02 : pool day with nhits >= min_duration (in the past)
        c02 = np.logical_and(is_pool_day, nhits >= min_duration)

        # case 03 : pool day with nhits + future_nhits >= min_duration (in the future)
        c03 = np.logical_and(is_pool_day, nhits + future_nhits >= min_duration)
        this_case[c03] = 3
        this_case[c02] = 2 # 2 is a subset of 3 and they have the same outcome, it is still helpful to keep track of it

        # case 04: pool day with nhits + future_nhits < min_duration (not an event)
        c04 = np.logical_and(is_pool_day, nhits + future_nhits < min_duration)
        this_case[c04] = 4

        # case 05: potentially a pool day (we don't know yet)
        c05 = maybe_pool_day
        this_case[c05] = 5

        # increase the counters for the spell (sintensity, sduration) in cases 12, 13, 14, 15 and 02, 03, 04, 05 (if count_with_pools)
        to_increase = np.logical_or(c13, c14) # 12 is subset of 13, 15 is subset of 14
        if count_with_pools:
            to_increase = np.logical_or.reduce([to_increase, c03, c04, c05]) # 02 is subset of 03
        sintensity[to_increase] += daily_intensity[i][to_increase]
        sduration[to_increase]  += 1

        # set the final intensity and duration for the events: only in cases 12, 13 and 02, 03 (if count_with_pools)
        to_set = c13 # 12 is subset of 13
        if count_with_pools:
            to_set = np.logical_or(to_set, c03) # 02 is subset of 03
        intensity[to_set] = sintensity[to_set]
        duration[to_set]  = sduration[to_set]
        
        # case 06: not hot day that is the end of a spell (reset all counters)
        c06 = np.logical_and.reduce([is_end_of_spell, np.logical_not(is_pool_day), np.logical_not(maybe_pool_day)])
        sintensity[c06] = 0
        sduration[c06]  = 0
        nhits[c06] = 0
        this_case[c06] = 6

        # the counters for the events (intensity, duration) are reset also for uncertain pool days (case 05), if pool_if_uncertain == 0
        if pool_if_uncertain == 0:
            c06 = np.logical_or(c06, c05)
        duration[c06]  = 0
        intensity[c06] = 0

    # once the loop is finished, prepare the output
    output = {
        'intensity' : intensity,
        'duration'  : duration,
        'interval'  : interval,
        'nhits'     : nhits,
        'sintensity': sintensity,
        'sduration' : sduration,
        'case'      : this_case,
    }
    
    return output

def calc_thresholds_cdo(data_nc: str,
                        thr_quantile: float,
                        window_size: int,
                        cdo_path: str = '/usr/bin/cdo',
                        var_name: str = 'data') -> Generator[xr.DataArray, None, None]:
    """
    Calculate thresholds using the Climate Data Operators (CDO) tool.
    This function computes thresholds based on a given quantile and window size 
    using CDO commands. It processes the input NetCDF data file to calculate 
    running minimum, running maximum, and the quantile-based thresholds. The 
    thresholds are returned as a generator of xarray DataArray objects.
    Parameters:
        data_nc (str): Path to the input NetCDF file containing the data.
        thr_quantile (float): Quantile value (between 0 and 1) used for threshold calculation.
        window_size (int): Size of the moving window for running calculations.
        cdo_path (str, optional): Path to the CDO executable. Defaults to '/usr/bin/cdo'.
        var_name (str, optional): Name of the variable in the NetCDF file to process. 
            Defaults to 'data'.
    Yields:
        xr.DataArray: Threshold values for each day of the year as an xarray DataArray. 
        The threshold corresponding to February 29th is calculated as the average of the
        thresholds for February 28th and March 1st.
    Notes:
        - The function uses subprocess to execute CDO commands, so the CDO tool 
            must be installed and accessible via the specified `cdo_path`.
        - Temporary files are created in the same directory as the input NetCDF file.
        - The function assumes the input data has a time dimension and processes 
            thresholds for each time step.
    """

    tmpdir = os.path.dirname(data_nc)

    # set the number of bins in CDO as an environment variable   
    data_ds = xr.open_dataset(data_nc)
    history_start = data_ds.time.min().values
    history_end = data_ds.time.max().values
    data_ds.close() 
    CDO_PCTL_NBINS = window_size * (history_end - history_start + 1) * 2 + 2
    os.environ['CDO_NUMBINS'] = str(CDO_PCTL_NBINS)

    # calculate running max and min of data using CDO
    datamin_nc = f'{tmpdir}/datamin.nc'
    datamax_nc = f'{tmpdir}/datamax.nc'

    cdo_cmd =  f'{cdo_path} ydrunmin,{window_size},rm=c {data_nc} {datamin_nc}'
    subprocess.run(cdo_cmd, shell = True)

    cdo_cmd = f'{cdo_path} ydrunmax,{window_size},rm=c {data_nc} {datamax_nc}'
    subprocess.run(cdo_cmd, shell = True)

    # calculate the thresholds
    threshold_quantile = int(thr_quantile*100)
    threshold_nc = f'{tmpdir}/threshold.nc'

    cdo_cmd = f'{cdo_path} ydrunpctl,{threshold_quantile},{window_size},rm=c,pm=r8 {data_nc} {datamin_nc} {datamax_nc} {threshold_nc}'
    subprocess.run(cdo_cmd, shell = True)

    # read the threshold data as Dataset
    thresholds = xr.open_dataset(threshold_nc)
    
    # extract the data as DataArray
    thresholds_da = thresholds[var_name]

    # add a 1-d "band" dimension
    thresholds_da = thresholds_da.expand_dims('band')
    
    # loop over the timesteps and yield the threshold
    days = thresholds_da.time.values
    for i, date in enumerate(days):
        time = dt.datetime.fromtimestamp(date.astype('O') / 1e9)
        if time.month == 2 and time.day == 29:
            # do the average between the thresholds for 28th of Feb and 1st of Mar
            threshold_data = (thresholds_da.isel(time = i-1) + thresholds_da.isel(time = i+1)) / 2
        else:
            threshold_data = thresholds_da.isel(time = i).drop_vars('time')

        yield threshold_data