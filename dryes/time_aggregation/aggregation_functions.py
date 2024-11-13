import xarray as xr
import datetime as dt
from typing import Callable, Optional
import numpy as np
import warnings

from functools import partial

from ..tools.timestepping import TimeRange, get_window
from ..tools.data import Dataset
from .time_aggregation import as_timagg_function

@as_timagg_function()
def average_of_window(size: int, unit: str,
                      propagate_metadata: Optional[str] = None) -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
    """
    def _average_of_window(variable: Dataset, time: dt.datetime, _size: int, _unit: str,
                           _propagate_metadata: Optional[str] = None) -> np.ndarray:
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
        """
        window = get_window(time, _size, _unit)

        if window.start < variable.get_first_ts().start:
            return None, {}
        
        time_signature = variable.time_signature
        if time_signature == 'end+1':
            window = TimeRange(window.start + dt.timedelta(days = 1), window.end + dt.timedelta(days = 1))

        #variable.make(window)
        times_to_get = variable.get_times(window)

        template = variable.build_templatearray(variable.get_template_dict())
        data_sum = np.zeros(template.shape, dtype = np.float64)
        valid_n = data_sum.copy()

        if _propagate_metadata is not None:
            metadata_list = []

        for time in times_to_get:
            new_data = variable.get_data(time)
            data_stack = np.stack([data_sum, new_data], axis = 0)
            data_sum = np.nansum(data_stack, axis = 0)
            valid_n += np.isfinite(new_data)

            if _propagate_metadata is not None:
                if _propagate_metadata in new_data.attrs:
                    this_metadata = new_data.attrs[_propagate_metadata]
                else:
                    this_metadata = ' '
                metadata_list.append(this_metadata)

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
    
    return partial(_average_of_window, _size = size, _unit = unit, _propagate_metadata = propagate_metadata)

# TODO: make sum safe for missing data
@as_timagg_function()
def sum_of_window(size: int, unit: str,
                  input_agg: Optional[dict] = None,
                  propagate_metadata: Optional[str] = None) -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using a sum over a certain period.
    input_agg expects a dictionary specifying the aggregation of the input data, with the following keys:
    - 'size':  int,  the size of the aggregation window.
    - 'unit':  str,  the unit of the aggregation window.
    - 'truncate_at_end_year': bool, whether to truncate the aggregation window at the end of the year or roll over.
    input_agg allows to pass if the input is already a sum, and discards some timesteps that are already included

    propagate_metadata: str, if not None, the metadata key to propagate from the input data to the output data
    """

    if input_agg is not None:
        if 'size' not in input_agg or 'unit' not in input_agg:
            raise ValueError('input_agg must have keys "size" and "unit"')
        if 'truncate_at_end_year' not in input_agg: input_agg['truncate_at_end_year'] = False

    def _sum_of_window(variable: Dataset, time: dt.datetime, _size: int, _unit: str,
                       _input_agg: Optional[dict] = None, _propagate_metadata: Optional[str] = None) -> np.ndarray:
        
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using a sum over a certain period.
        """
        window = get_window(time, _size, _unit)
        if window.start < variable.get_first_ts().start:
            return None, {}
        
        #variable.make(window)
        time_signature = variable.time_signature
        if time_signature == 'end+1':
            window = TimeRange(window.start + dt.timedelta(days = 1), window.end + dt.timedelta(days = 1))

        all_times = variable.get_times(window)
        if _input_agg is not None:
            if 'end' in time_signature:
                all_times.sort(reverse=True)
            else:
                all_times.sort()
            all_times_loop = all_times.copy()
            for ttime in all_times_loop:
                if ttime not in all_times:
                    continue
                if time_signature == 'start':
                    this_time_window = get_window(ttime, _input_agg['size'], _input_agg['unit'], start = True)
                    if _input_agg['truncate_at_end_year'] and this_time_window.end.year != ttime.year:
                        this_time_window = TimeRange(this_time_window.start, dt.datetime(ttime.year, 12, 31))
                    
                    included_times = [t for t in all_times if this_time_window.start < t <= this_time_window.end]
                else:
                    if time_signature == 'end+1':
                        ttime = ttime - dt.timedelta(days = 1)

                    this_time_window = get_window(ttime, _input_agg['size'], _input_agg['unit'], start = False)
                    if _input_agg['truncate_at_end_year'] and this_time_window.start.year != ttime.year:
                        this_time_window = TimeRange(dt.datetime(ttime.year, 1, 1), this_time_window.end)

                    if time_signature == 'end+1':
                        included_times = [t for t in all_times if this_time_window.start <= t - dt.timedelta(days = 1) < this_time_window.end]
                        breakpoint()
                    elif time_signature == 'end':
                        included_times = [t for t in all_times if this_time_window.start <= t < this_time_window.end]

                for t in included_times:
                    all_times.remove(t)
                    
        all_times.sort()
        times_to_get = all_times

        template = variable.build_templatearray(variable.get_template_dict())
        data = np.zeros(template.shape, dtype = np.float64)

        if _propagate_metadata is not None:
            metadata_list = []
        for ttime in times_to_get:
            new_data = variable.get_data(ttime)
            data_stack = np.stack([data, new_data], axis = 0)
            data = np.sum(data_stack, axis = 0)
            if _propagate_metadata is not None:
                if _propagate_metadata in new_data.attrs:
                    this_metadata = new_data.attrs[_propagate_metadata]
                else:
                    this_metadata = ' '
                metadata_list.append(this_metadata)

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
        if _propagate_metadata is not None:
            agg_info[_propagate_metadata] = ','.join(metadata_list)

        return data, agg_info
    
    return partial(_sum_of_window, _size = size, _unit = unit, _input_agg = input_agg, _propagate_metadata = propagate_metadata)

@as_timagg_function()
def weighted_average_of_window(size: int, unit: str, input_agg: str|dict, weights = 'overlap') -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
    weights can be 'overlap' or 'inv_distance' (to end) 
    input_agg expects a dictionary specifying the aggregation of the input data, with the following keys:
        - 'size':  int,  the size of the aggregation window.
        - 'unit':  str,  the unit of the aggregation window.
        - 'truncate_at_end_year': bool, whether to truncate the aggregation window at the end of the year or roll over.
        [for VIIRS data: input_agg = 'viirs' means {'size': 8, 'unit': 'days', 'truncate_at_end_year': True}]
    """

    if isinstance(input_agg, str) and input_agg == 'viirs':
        input_agg = {'size': 8, 'unit': 'days', 'truncate_at_end_year': True}
    elif isinstance(input_agg, dict):
        if 'size' not in input_agg or 'unit' not in input_agg:
            raise ValueError('input_agg must have keys "size" and "unit"')
        if 'truncate_at_end_year' not in input_agg: input_agg['truncate_at_end_year'] = False

    if weights not in ['overlap', 'inv_distance']:
        raise ValueError('weights must be either "overlap" or "inv_distance"')

    def _weighted_average_of_window(variable: Dataset, time: dt.datetime,
                                    _size: int, _unit: str, _input_agg: dict, _weights = 'overlap') -> np.ndarray:
        """
        Aggregates the FAPAR data to the current timestep using inverse time distance of the previous 
        """
        window = get_window(time, _size, _unit)
        start = window.start
        end = window.end
        
        # get the window of the relevant data (this is the range of times that we should search the data for the cover the window)
        input_time_signature = variable.time_signature
        if input_time_signature == 'start':
            data_start = window.start - dt.timedelta(**{input_agg['unit']: input_agg['size'] - 1})
            data_end = window.end
        elif input_time_signature == 'end':
            data_start = window.start
            data_end = window.end + dt.timedelta(**{input_agg['unit']: input_agg['size'] - 1})
        elif input_time_signature == 'end+1':
            data_start = window.start + dt.timedelta(days = 1)
            data_end = window.end + dt.timedelta(**{input_agg['unit']: input_agg['size']})

        available_data = variable.get_times(TimeRange(data_start, data_end))
        if len(available_data) == 0:
            return None, {}

        overlap = []
        distance = []
        for data_time in available_data:
            if input_time_signature == 'start':
                data_start = data_time
                data_end = data_time + dt.timedelta(**{input_agg['unit']: input_agg['size'] - 1})
                if _input_agg['truncate_at_end_year'] and data_end.year != data_start.year:
                    data_end = dt.datetime(data_start.year, 12, 31)
            else:
                if input_time_signature == 'end+1':
                    data_end = data_time - dt.timedelta(days = 1)
                elif input_time_signature == 'end':
                    data_end = data_time
                data_start = data_end - dt.timedelta(**{input_agg['unit']: input_agg['size'] - 1})
                if _input_agg['truncate_at_end_year'] and data_start.year != data_end.year:
                    data_start = dt.datetime(data_end.year, 1, 1)

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