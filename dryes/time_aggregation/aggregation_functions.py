import xarray as xr
import datetime as dt
from typing import Callable, Optional
import numpy as np
import warnings

from functools import partial

from ..io import IOHandler
from ..tools.timestepping import TimeRange, get_window

def average_of_window(size: int, unit: str) -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
    """
    def _average_of_window(variable: IOHandler, time: dt.datetime, _size: int, _unit: str) -> np.ndarray:
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
        """
        window = get_window(time, _size, _unit)

        if window.start < variable.start:
            return None, {}

        #variable.make(window)
        times_to_get = variable.get_times(window)

        data_sum = variable.get_data(times_to_get[0])
        valid_n = np.isfinite(data_sum).astype(int)

        for time in times_to_get[1:]:
            this_data = variable.get_data(time)
            data_stack = np.stack([data_sum, this_data], axis = 0)
            data_sum = np.nansum(data_stack, axis = 0)
            valid_n += np.isfinite(this_data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_data = data_sum / valid_n

        # data = [variable.get_data(time) for time in times_to_get]
        # all_data = np.stack(data, axis = 0)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=RuntimeWarning)
        #     mean_data = np.nanmean(all_data, axis = 0)
        # mean = variable.template.copy(data = mean_data)
        # mean = mean.assign_coords(time = time)
        
        unit_singular = unit[:-1] if unit[-1] == 's' else unit
        su_string = f'{_size} {unit_singular}{"" if _size == 1 else "s"}'
        agg_info = {'agg_type' : f'average, {su_string}',
                    'agg_start': f'{min(times_to_get):%Y-%m-%d}',
                    'agg_end'  : f'{max(times_to_get):%Y-%m-%d}',
                    'agg_n'    : len(times_to_get)}

        return mean_data, agg_info
    
    return partial(_average_of_window, _size = size, _unit = unit)

# TODO: make sum safe for missing data
def sum_of_window(size: int, unit: str, input_agg: Optional[dict] = None) -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using a sum over a certain period.
    input_agg expects a dictionary specifying the aggregation of the input data, with the following keys:
    - 'size':  int,  the size of the aggregation window.
    - 'unit':  str,  the unit of the aggregation window.
    - 'start': bool, whether the timestep is the start or the end of the aggregation window.
    - 'truncate_at_end_year': bool, whether to truncate the aggregation window at the end of the year or roll over.
    input_agg allows to pass if the input is already a sum, and discards some timesteps that are already included
    """

    if input_agg is not None:
        if 'size' not in input_agg or 'unit' not in input_agg:
            raise ValueError('input_agg must have keys "size" and "unit"')
        if 'start' not in input_agg: input_agg['start'] = False
        if 'truncate_at_end_year' not in input_agg: input_agg['truncate_at_end_year'] = False

    def _sum_of_window(variable: IOHandler, time: dt.datetime, _size: int, _unit: str,
                       _input_agg: Optional[dict] = None) -> np.ndarray:
        
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using a sum over a certain period.
        """
        window = get_window(time, _size, _unit)
        if window.start < variable.start:
            return None, {}
        
        #variable.make(window)
        all_times = variable.get_times(window)
        if _input_agg is not None:
            all_times.sort(reverse=True)
            all_times_loop = all_times.copy()
            for time in all_times_loop:
                if time not in all_times:
                    continue
                if _input_agg['start']:
                    this_time_window = get_window(time, _input_agg['size'], _input_agg['unit'], start = True)
                    if _input_agg['truncate_at_end_year'] and this_time_window.end.year != time.year:
                        this_time_window = TimeRange(this_time_window.start, dt.datetime(time.year, 12, 31))
                else:
                    this_time_window = get_window(time, _input_agg['size'], _input_agg['unit'], start = False)
                    if _input_agg['truncate_at_end_year'] and this_time_window.start.year != time.year:
                        this_time_window = TimeRange(dt.datetime(time.year, 1, 1), this_time_window.end)
                included_times = [t for t in all_times if this_time_window.start <= t < this_time_window.end]
                for t in included_times:
                    all_times.remove(t)

        all_times.sort()
        times_to_get = all_times

        data = variable.get_data(times_to_get[0])
        for time in times_to_get[1:]:
            data_stack = np.stack([data, variable.get_data(time)], axis = 0)
            data = np.sum(data_stack, axis = 0)

        # data = [variable.get_data(time) for time in times_to_get]
        # all_data = np.stack(data, axis = 0)
        # sum_data = np.sum(all_data, axis = 0)
        # sum = variable.template.copy(data = sum_data)
        # sum = sum.assign_coords(time = time)
        unit_singular = unit[:-1] if unit[-1] == 's' else unit
        su_string = f'{_size} {unit_singular}{"" if _size == 1 else "s"}'
        agg_info = {'agg_type' : f'sum, {su_string}',
                    'agg_start': f'{min(times_to_get):%Y-%m-%d}',
                    'agg_end'  : f'{max(times_to_get):%Y-%m-%d}',
                    'agg_n'    : len(times_to_get)}

        return data, agg_info
    
    return partial(_sum_of_window, _size = size, _unit = unit, _input_agg = input_agg)

def weighted_average_of_window(size: int, unit: str, input_agg: str|dict, weights = 'overlap') -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
    weights can be 'overlap' or 'inv_distance' (to end) 
    input_agg expects a dictionary specifying the aggregation of the input data, with the following keys:
        - 'size':  int,  the size of the aggregation window.
        - 'unit':  str,  the unit of the aggregation window.
        - 'start': bool, whether the timestep is the start or the end of the aggregation window.
        - 'truncate_at_end_year': bool, whether to truncate the aggregation window at the end of the year or roll over.
        [for VIIRS data: input_agg = 'viirs' means {'size': 8, 'unit': 'days', 'start': True, 'truncate_at_end_year': True}]
    """

    if isinstance(input_agg, str) and input_agg == 'viirs':
        input_agg = {'size': 8, 'unit': 'days', 'start': True, 'truncate_at_end_year': True}
    elif isinstance(input_agg, dict):
        if 'size' not in input_agg or 'unit' not in input_agg:
            raise ValueError('input_agg must have keys "size" and "unit"')
        if 'start' not in input_agg: input_agg['start'] = False
        if 'truncate_at_end_year' not in input_agg: input_agg['truncate_at_end_year'] = False

    if weights not in ['overlap', 'inv_distance']:
        raise ValueError('weights must be either "overlap" or "inv_distance"')

    def _weighted_average_of_window(variable: IOHandler, time: dt.datetime,
                                    _size: int, _unit: str, _input_agg: dict, _weights = 'overlap') -> np.ndarray:
        """
        Aggregates the FAPAR data to the current timestep using inverse time distance of the previous 
        """
        end    = time                                    # this is where the current period ends
        window = get_window(time, _size, _unit)
        start  = window.start                            # this is where the averaging period starts

        # get the window of the relevant data (this is the range of times that we should search the data for the cover the window)
        if _input_agg['start']:
            data_start = window.start - dt.timedelta(**{input_agg['unit']: input_agg['size'] - 1})
            data_end = window.end
        else:
            data_start = window.start
            data_end = window.end + dt.timedelta(**{input_agg['unit']: input_agg['size'] - 1})

        available_data = variable.get_times(TimeRange(data_start, data_end))
        if len(available_data) == 0:
            return variable.template.values, {}

        overlap = []
        distance = []
        for data_time in available_data:
            if _input_agg['start']:
                data_start = data_time
                data_end = data_time + dt.timedelta(**{input_agg['unit']: input_agg['size'] - 1})
                if _input_agg['truncate_at_end_year'] and data_end.year != data_time.year:
                    data_end = dt.datetime(data_time.year, 12, 31)
            else:
                data_end = data_time
                data_start = data_time - dt.timedelta(**{input_agg['unit']: input_agg['size'] - 1})
                if _input_agg['truncate_at_end_year'] and data_start.year != data_time.year:
                    data_start = dt.datetime(data_time.year, 1, 1)

            distance.append(abs((end - data_end).days) + 1)
            overlap.append((min(end, data_end) - max(start, data_start)).days + 1)

        # open all the data
        databrick = np.stack([variable.get_data(t).values for t in available_data], axis = 0)

        # calculate the weights
        if _weights == 'overlap':
            weights = np.array(overlap)
        elif _weights == 'inv_distance':
            weights = 1/np.array(distance)

        # multiply each data in the brick by the weight, b is the band of the raster (which is always 1)
        weighted_data = np.einsum('i,ibjk->ibjk', weights, databrick)
        
        # sum all the weighted data and divide by the total weight, protecting nans
        mask = np.isfinite(weighted_data) # <- this is a mask of all the nans in the weighted data
        weights_3d = np.broadcast_to(weights[:, np.newaxis, np.newaxis, np.newaxis], weighted_data.shape)
        selected_weights = np.where(mask, weights_3d, 0)
        total_weights = np.sum(selected_weights, axis=0)

        # sum all the non-nan weighted data
        weighted_sum = np.nansum(weighted_data, axis = 0)
        
        # divide by the total weights
        # to avoid division by zero, we add a small number to the total weights, these points will be masked anyways
        total_weights = np.where(total_weights < 1e-3, 1e-3, total_weights)
        weighted_mean = weighted_sum / total_weights

        # Remove where the total sum of weights is less than half the total weights
        weighted_mean = np.where(total_weights/sum(weights) < 0.5, np.nan, weighted_mean)
        #weighted_mean = np.where(total_weights < 0.2, np.nan, weighted_mean)

        timesteps_str = ', '.join([t.strftime('%Y-%m-%d') for t in available_data])
        weights_str   = ', '.join([f'{w:.2f}' for w in weights])

        agg_method_str = f'weighted ({_weights}) average, {_size} {_unit}'

        agg_info = {'agg_method': agg_method_str,
                    'agg_timesteps': timesteps_str,
                    'agg_n_timesteps': len(available_data),
                    'agg_weights': weights_str}

        return weighted_mean, agg_info

    return partial(_weighted_average_of_window, _size = size, _unit = unit, _input_agg = input_agg, _weights = weights)