import xarray as xr
from datetime import datetime
from typing import Callable
import numpy as np
import warnings

from functools import partial

from ..variables.dryes_variable import DRYESVariable
from ..lib.time import get_window
from ..lib.space import Grid
from ..lib.io import get_data, check_data_range

def average_of_window(size: int, unit: str) -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
    """
    def _average_of_window(variable: DRYESVariable, grid: Grid, time: datetime, _size: int, _unit: str) -> xr.DataArray:
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
        """
        window = get_window(time, _size, _unit)

        if window.start < variable.start:
            return None

        variable.make(grid, window)
        times_to_get = check_data_range(variable.path, window)

        data = [get_data(variable.path, time) for time in times_to_get]
        all_data = np.stack(data, axis = 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_data = np.nanmean(all_data, axis = 0)
        mean = grid.template.copy(data = mean_data)
        mean = mean.assign_coords(time = time)
        return mean
    
    return partial(_average_of_window, _size = size, _unit = unit)

def sum_of_window(size: int, unit: str) -> Callable:
    """
    Returns a function that aggregates the data in a DRYESDataset at the timestep requested, using a sum over a certain period.
    """
    def _sum_of_window(variable: DRYESVariable, grid: Grid, time: datetime, _size: int, _unit: str) -> xr.DataArray:
        """
        Aggregates the data in a DRYESDataset at the timestep requested, using an average over a certain period.
        """
        window = get_window(time, _size, _unit)
        if window.start < variable.start:
            return None

        variable.make(grid, window)
        times_to_get = check_data_range(variable.path, window)

        data = [get_data(variable.path, time) for time in times_to_get]
        all_data = np.stack(data, axis = 0)
        sum_data = np.sum(all_data, axis = 0)
        sum = grid.template.copy(data = sum_data)
        sum = sum.assign_coords(time = time)
        return sum
    
    return partial(_sum_of_window, _size = size, _unit = unit)