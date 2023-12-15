import rioxarray
import xarray as xr
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Callable

from functools import partial

from ..variables.dryes_variable import DRYESVariable
from ..lib.time import TimeRange, get_interval_date, get_window
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
        data = xr.concat(data, dim = 'time')
        mean = data.mean(dim = 'time', skipna=True)
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
        data = xr.concat(data, dim = 'time')
        mean = data.sum(dim = 'time', skipna=True)
        mean = mean.assign_coords(time = time)
        return mean
    
    return partial(_sum_of_window, _size = size, _unit = unit)