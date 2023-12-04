import rioxarray
import xarray as xr
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Callable

from functools import partial

from ..variables.dryes_variable import DRYESVariable
from ..lib.time import TimeRange, get_interval_date
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
        #breakpoint()
        time_end = time - timedelta(days=1)
        if _unit[-1] != 's': _unit += 's'
        if _unit in ['months', 'years', 'days', 'weeks']:
            time_start = time - relativedelta(**{_unit: _size})
        elif _unit == 'dekads':
            if time_end == get_interval_date(time_end, 36, end = True):
                tmp_time = time_end
                for i in range(_size): tmp_time = get_interval_date(tmp_time, 36) - timedelta(days=1)
                time_start = tmp_time + timedelta(days=1)
            else:
                time_start = time - timedelta(days = 10 * _size)
        else:
            raise ValueError('Unit for average aggregator recognized: must be one of dekads, months, years, days, weeks')

        if time_start < variable.start:
            return None

        variable.make(grid, TimeRange(time_start, time_end))
        times_to_get = check_data_range(variable.path, TimeRange(time_start, time_end))

        data = [get_data(variable.path, time) for time in times_to_get]
        data = xr.concat(data, dim = 'time')
        mean = data.mean(dim = 'time', skipna=True)
        mean = mean.assign_coords(time = time)
        return mean
    
    return partial(_average_of_window, _size = size, _unit = unit)