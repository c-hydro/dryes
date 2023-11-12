import datetime
import xarray as xr
import os

from typing import Optional

from ..data_processes import Grid, TimeSteps, TimeRange
from ..lib.log import log

class DRYESDataSource:

    def __init__(self) -> None:
        """
        Creates a DRYESDataSource object.
        """
        self.isstatic = False
    
    def get_data(self, space: Grid, time: Optional[datetime.datetime] = None) -> xr.Dataset:
        """
        Get data from the data source as an xarray.Dataset.
        for a single time. This is a mandatory method for all subclasses.
        """
        raise NotImplementedError

    def get_times(time_start: datetime.datetime, time_end: datetime.datetime) -> datetime.datetime:
        """
        Get a list of times between two dates.
        This is a mandatory method for all subclasses.
        """
        raise NotImplementedError

    def get_timerange(self) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Get the timerange of validity of the data source.
        This is a mandatory method for all subclasses.
        """
        raise NotImplementedError
    
    def make_local(self, grid: Grid, timesteps: Optional[TimeSteps|TimeRange], name: str, path: str):
        """
        Saves data from the data source to a geotiff.
        """

        source_name = self.__class__.__name__
        variable_name = self.variable
        log(f'Starting download of {variable_name} using {source_name}.')

        if not self.isstatic:
            if timesteps is None:
                timesteps = TimeRange(self.get_timerange())
            if timesteps.isrange:
                timesteps = [s for s in self.get_times(timesteps.start, timesteps.end)]
            else:
                available = self.get_timerange()
                timesteps = [s for s in timesteps.timesteps if s >= available[0] and s <= available[1]]
                start = min(timesteps)
                end = max(timesteps)
                timesteps = [s for s in timesteps if s in self.get_times(start, end)]

            log(f'Found {len(timesteps)} timesteps between {min(timesteps):%Y-%m-%d} and {max(timesteps):%Y-%m-%d}.')
        else:
            if timesteps is not None:
                log(f'Warning: {variable_name} from {source_name} is a static data. Ignoring timesteps.')
            timesteps = [None]

        for i, time in enumerate(timesteps):
            if time is not None:
                log(f'Starting {time:%Y-%m-%d} ({i+1} of {len(timesteps)}).')

            data = self.get_data(grid, time)

            for i, var in enumerate(data.data_vars):
                if time is None:
                    output_path = os.path.join(path, '{name}_{var}.tif')
                    output_path = output_path.format(name = name, var = var)
                else:
                    output_path = os.path.join(path, '{name}_{var}_{time:%Y%m%d}.tif')
                    output_path = output_path.format(name = name, var = var, time = time)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                data[var].rio.to_raster(output_path, compress = 'lzw')
            
            log(f'Saved {i + 1} files to {os.path.dirname(output_path)}')