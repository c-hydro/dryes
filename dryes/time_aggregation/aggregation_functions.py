import xarray as xr
from datetime import datetime
from typing import Callable
import numpy as np
import warnings

from functools import partial

from ..io import IOHandler
from ..utils.time import get_window

def average_of_window(size: int, unit: str) -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
    """
    def _average_of_window(variable: IOHandler, time: datetime, _size: int, _unit: str) -> np.ndarray:
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
        """
        window = get_window(time, _size, _unit)

        if window.start < variable.start:
            return None

        #variable.make(window)
        times_to_get = variable.get_times(window)

        data_sum = variable.get_data(times_to_get[0])
        valid_n  = np.isfinite(data_sum)
        for time in times_to_get[1:]:
            this_data = variable.get_data(time)
            data_stack = np.stack([data_sum, this_data], axis = 0)
            data_sum = np.nansum(data_stack, axis = 0)
            valid_n += np.isfinite(this_data)

        mean_data = data_sum / valid_n

        # data = [variable.get_data(time) for time in times_to_get]
        # all_data = np.stack(data, axis = 0)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=RuntimeWarning)
        #     mean_data = np.nanmean(all_data, axis = 0)
        # mean = variable.template.copy(data = mean_data)
        # mean = mean.assign_coords(time = time)
        
        agg_info = {'agg_type': 'average',
                    'agg_start': f'{min(times_to_get):%Y-%m-%d}',
                    'agg_end':   f'{max(times_to_get):%Y-%m-%d}',
                    'agg_n': len(times_to_get)}

        return mean_data, agg_info
    
    return partial(_average_of_window, _size = size, _unit = unit)

# TODO: make sum safe for missing data
def sum_of_window(size: int, unit: str) -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using a sum over a certain period.
    """
    def _sum_of_window(variable: IOHandler, time: datetime, _size: int, _unit: str) -> np.ndarray:
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using a sum over a certain period.
        """
        window = get_window(time, _size, _unit)
        if window.start < variable.start:
            return None
        
        #variable.make(window)
        times_to_get = variable.get_times(window)

        data = variable.get_data(times_to_get[0])
        for time in times_to_get[1:]:
            data_stack = np.stack([data, variable.get_data(time)], axis = 0)
            data = np.nansum(data_stack, axis = 0)

        # data = [variable.get_data(time) for time in times_to_get]
        # all_data = np.stack(data, axis = 0)
        # sum_data = np.sum(all_data, axis = 0)
        # sum = variable.template.copy(data = sum_data)
        # sum = sum.assign_coords(time = time)

        agg_info = {'agg_type': 'sum',
                    'agg_start': f'{min(times_to_get):%Y-%m-%d}',
                    'agg_end':   f'{max(times_to_get):%Y-%m-%d}',
                    'agg_n': len(times_to_get)}

        return data, agg_info
    
    return partial(_sum_of_window, _size = size, _unit = unit)