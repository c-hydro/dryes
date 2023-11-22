from datetime import datetime, timedelta
from functools import partial
import xarray as xr
import rioxarray as rxr
import os

from typing import Optional


from ...lib.space import Grid

class LocalSource:

    def __init__(self, path_pattern: str, varname: Optional[str] = None):
        self.path_pattern = path_pattern
        if varname is None:
            self.variable = 'variable'
        else:
            self.variable = varname
        
        # list of preprocess algorithms to be applied to the data
        self._preprocess = []

    def get_data(self, space: Optional[Grid] = None, time: Optional[datetime] = None) -> xr.Dataset:
        """
        Get data from the local data source as an xarray.Dataset.
        """
        if time is None:
            if self.path_is_dynamic():
                raise ValueError('Dynamic data source requires a time argument.')
            else:
                path = self.path_pattern
        else:
            path = time.strftime(self.path_pattern)
        
        data = rxr.open_rasterio(path)
        if space is not None:
            data = space.apply(data)

        return xr.Dataset({self.variable: data})
    
    def path_is_dynamic(self):
        """
        Check if the path pattern contains any wildcard making it dynamic.
        """
        available_wildcards = ['%Y', '%m', '%d', '%j']
        return any(wc in self.path_pattern for wc in available_wildcards)

    def check_data(self, time: datetime) -> bool:
        """
        Check if data is available for a given time.
        """
        path = time.strftime(self.path_pattern)
        return os.path.exists(path)

    def get_times(self, time_start: datetime, time_end: datetime) -> datetime:
        """
        Get a list of times between two dates.
        """
        time = time_start
        while time <= time_end:
            if self.check_data(time):
                yield time
            time += timedelta(days=1)

    def get_start(self) -> datetime:
        """
        Get the start of the available data.
        """
        time_start = datetime(1900, 1, 1)
        time_end = datetime.now()
        for time in self.get_times(time_start, time_end):
            return time