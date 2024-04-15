import xarray as xr
import datetime as dt
from typing import Callable, Optional
import numpy as np
import warnings

from functools import partial

from ..io import IOHandler
from ..utils.time import get_window, get_interval_date, TimeRange

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
            return None

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

    if 'size' not in input_agg or 'unit' not in input_agg:
        raise ValueError('input_agg must have keys "size" and "unit"')
    if 'start' not in input_agg: input_agg['start'] = False
    if 'truncate_at_end_year' not in input_agg: input_agg['truncate_at_end_year'] = False

    def _sum_of_window(variable: IOHandler, time: dt.datetime, _size: int, _unit: str,
                       _input_agg: Optional[tuple[int, str]] = None) -> np.ndarray:
        
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using a sum over a certain period.
        """
        window = get_window(time, _size, _unit)
        if window.start < variable.start:
            return None
        
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