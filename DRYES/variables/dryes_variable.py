import numpy as np
from datetime import datetime, timedelta
import xarray as xr
import rioxarray

import os
from typing import Callable, Optional, Iterator

from ..data_sources import DRYESDataSource
from ..lib.log import log
from ..lib.io import save_dataset_to_geotiff, save_dataarray_to_geotiff
from ..lib.time import TimeRange
from ..lib.space import Grid

class DRYESVariable:
    def __init__(self, variable_name: str, type: str, destination: str) -> None:

        self.name = variable_name
        self.isstatic = type == 'static'
        self.destination = destination
        self.addn = '_%Y%m%d' if not self.isstatic else ''
        self.path_pattern = os.path.join(self.destination, self.name+self.addn+'.tif')

    @staticmethod
    def from_data_source(data_source: DRYESDataSource, variable: str, type: str, destination: str) -> 'DRYESVarFromDataSource':
        return DRYESVarFromDataSource(data_source, variable, type, destination)

    @staticmethod
    def from_other_variables(variables: list[str], function: Callable[..., np.ndarray], name:str, type: str, destination:str) -> 'DRYESVarFromOtherVars':
        return DRYESVarFromOtherVars(variables, function, name, type, destination)
    
    def get_data(self, time: Optional[datetime] = None) -> xr.DataArray:
        path = time.strftime(self.path_pattern) if not self.isstatic else self.path_pattern
        data = rioxarray.open_rasterio(path)
        return data
    
    def check_data(self, time: Optional[datetime] = None) -> bool:
        path = time.strftime(self.path_pattern) if time is not None else self.path_pattern
        return os.path.isfile(path)
    
    def check_data_range(self, time_range: TimeRange) -> bool|Iterator[datetime]:
        """
        Yields the timesteps in the TimeRange that are available locally
        """
        if self.isstatic:
            return self.check_data()
        
        time = time_range.start
        while time <= time_range.end:
            if self.check_data(time):
                yield time
            time += timedelta(days=1)

class DRYESVarFromDataSource(DRYESVariable):
    def __init__(self, data_source: DRYESDataSource, variable: str, type: str, destination: str) -> None:
        self.data_source = data_source
        self.start = data_source.get_start()
        super().__init__(variable, type, destination)

    def gather(self, grid: Grid, time_range: Optional[TimeRange] = None) -> None:
        """
        Gathers all the data from the remote source in the TimeRange,
        also checks that the data is not available yet before gathering it
        """

        source = self.data_source
        source_name = source.__class__.__name__
        variable_name = source.variable
        log(f'Starting download of {variable_name} using {source_name}.')

        if time_range is None and not self.isstatic:
            raise ValueError(f'TimeRange must be specified for dynamic data: {variable_name}.')
        
        if self.isstatic:
            # check if the data is already available locally
            if self.check_data():
                log(f'Data already available locally.')
                return

            # download the data
            log(f'Downloading static data {variable_name}.')
            data = source.get_data(grid)
            output_path = self.destination
            n_saved = save_dataset_to_geotiff(data, output_path)

            log(f'Saved {n_saved} files to {output_path}')
        
        else:
            # get all the timesteps that the source is available for in the timerange
            timesteps_to_download = list(source.get_times(time_range.start, time_range.end))
            tot_timesteps = len(timesteps_to_download)
            log(f'Found {tot_timesteps} timesteps between {time_range.start:%Y-%m-%d} and {time_range.end:%Y-%m-%d}.')

            # filter out the timesteps that are already available locally
            timesteps_to_download = [time for time in timesteps_to_download if not self.check_data(time)]
            num_timesteps = len(timesteps_to_download)
            if num_timesteps == 0:
                log(f'All timesteps already available locally.')
                return
            
            log(f'Found {num_timesteps} timesteps not already available locally.')

            # download each remaining timestep
            for time in timesteps_to_download:
                log(f'Downloading {variable_name} for {time:%Y-%m-%d}.')
                data = source.get_data(grid, time)
                output_path = self.destination.format(time = time)
                addn = self.addn.format(time = time)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                data.rio.to_raster(output_path, compress = 'lzw')
                n_saved = save_dataset_to_geotiff(data, output_path, addn)
        
                log(f'Saved {n_saved} files to {output_path}')

class DRYESVarFromOtherVars(DRYESVariable):
    def __init__(self, variables: dict[str:DRYESVariable], function: Callable[..., xr.DataArray], name: str, type: str, destination: str) -> None:
        self.variables = variables
        self.function = function
        self.start = max([v.start for v in variables.values() if not v.isstatic])
        super().__init__(name, type, destination)

    def compute(self, time_range: Optional[TimeRange]):
        """
        Computes the data from the other variables in the TimeRange,
        also checks that the data is not available yet before computing it
        """
        
        variable_name = self.name
        log(f'Starting computation of {variable_name} data.')

        if time_range is None and not self.isstatic:
            raise ValueError(f'TimeRange must be specified for dynamic data: {variable_name}.')
        
        # make sute the static variables are available
        static_vars = [v for v in self.variables.values() if v.isstatic]
        check_static = [v.check_data() for v in static_vars]
        if not all(check_static):
            missing = [v.variable_name for i, v in enumerate(static_vars) if not check_static[i]]
            log(f'Static variables {missing.join(', ')} are not available.')
            return

        if self.isstatic:
            # check if the data is already available locally
            if self.check_data():
                log(f'Data already available locally.')
                return

            # calculate the data
            log(f'Computing static data {variable_name}.')
            data = self.function(**{k:v.get_data().values for k,v in self.variables.items()})
            output_file = self.path_pattern
            saved = save_dataarray_to_geotiff(data, output_file)

            log(f'Saved to {output_file}')
        
        else:
            # get all the timesteps where we have access to all of the dynamic variables
            dynamic_vars = [v for v in self.variables.values() if not v.isstatic]
            timesteps_to_compute_per_var = [set(v.check_data_range(time_range)) for v in dynamic_vars]
            intersection = set.intersection(*timesteps_to_compute_per_var)
            timesteps_to_compute = list(intersection)
            timesteps_to_compute.sort() # sort the timesteps in chronological order <- going through the set messes up the order
            tot_timesteps = len(timesteps_to_compute)
            log(f'Found {tot_timesteps} timesteps between {time_range.start:%Y-%m-%d} and {time_range.end:%Y-%m-%d}.')

            # filter out the timesteps that are already computed
            timesteps_to_compute = [time for time in timesteps_to_compute if not self.check_data(time)]
            num_timesteps = len(timesteps_to_compute)
            if num_timesteps == 0:
                log(f'All timesteps already computed.')
                return
            
            log(f'Found {num_timesteps} timesteps not already computed.')

            # get the static variables, these are the same for each timestep
            static_data = {k:v.get_data() for k,v in self.variables.items() if v.isstatic}

            # compute each remaining timestep
            for time in timesteps_to_compute:
                log(f'Computing {variable_name} for {time:%Y-%m-%d}.')
                dynamic_data = {k:v.get_data(time) for k,v in self.variables.items() if not v.isstatic}
                data = self.function(**static_data, **dynamic_data)
                output_file = time.strftime(self.path_pattern)
                saved = save_dataarray_to_geotiff(data, output_file)

                log(f'Saved to {output_file}')