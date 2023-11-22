from __future__ import annotations

import xarray as xr

from typing import Callable, Optional

from ..lib.log import log
from ..lib.io import save_dataarray_to_geotiff, check_data, check_data_range, get_data
from ..lib.time import TimeRange
from ..lib.space import Grid

from . import DRYESInput

class DRYESVariable():
    def __init__(self, inputs: dict[str:DRYESInput],\
                       grid_file: str,
                       function: Callable[..., xr.DataArray],
                       destination: str,
                       name: Optional[str]=None) -> None:
        
        self.inputs = inputs
        self.start = max([v.start for v in inputs.values() if not v.isstatic])
        self.grid = Grid(grid_file)

        self.function = function
        
        if name is None: name = 'variable'
        self.name = name
        self.path = destination

    def gather_inputs(self, time_range: TimeRange) -> None:
        """
        Gathers all the data from the remote source in the TimeRange,
        also checks that the data is not available yet before gathering it
        """

        log('Gathering input data...')
        for v in self.inputs.values():
            v.gather(self.grid, time_range)

    def compute(self, time_range: TimeRange):
        """
        Computes the data from the other inputs in the TimeRange,
        also checks that the data is not available yet before computing it
        """
        
        variable_name = self.name
        log(f'Starting {variable_name} computation...')

        variable_paths = [v.path for v in self.inputs.values()]

        timesteps_to_compute_per_var = [set(check_data_range(paths, time_range)) for paths in variable_paths]
        intersection = set.intersection(*timesteps_to_compute_per_var)
        timesteps_to_compute = list(intersection)
        timesteps_to_compute.sort() # sort the timesteps in chronological order <- going through the set messes up the order
        tot_timesteps = len(timesteps_to_compute)
        log(f'Found {tot_timesteps} timesteps between {time_range.start:%Y-%m-%d} and {time_range.end:%Y-%m-%d}.')

        # filter out the timesteps that are already computed
        timesteps_to_compute = [time for time in timesteps_to_compute if not check_data(self.path, time)]
        num_timesteps = len(timesteps_to_compute)
        if num_timesteps == 0:
            log(f'All timesteps already computed.')
            return
        
        log(f'Found {num_timesteps} timesteps not already computed.')

        # get the static inputs, these are the same for each timestep
        static_data = {k:get_data(v.path_pattern) for k,v in self.inputs.items() if v.isstatic}

        # compute each remaining timestep
        for time in timesteps_to_compute:
            log(f'Computing {variable_name} for {time:%Y-%m-%d}.')
            dynamic_data = {k:get_data(v.path_pattern, time) for k,v in self.inputs.items() if not v.isstatic}
            data = self.function(**static_data, **dynamic_data)
            output_file = time.strftime(self.path)
            saved = save_dataarray_to_geotiff(data, output_file)

            log(f'Saved to {output_file}')

    def make(self, time_range: TimeRange) -> None:
        """
        Gathers the data from the remote source in the TimeRange
        and preprocesses it using the function
        """

        self.gather_inputs(time_range)
        self.compute(time_range)

    @staticmethod
    def identical(input: DRYESInput, grid_file: str) -> DRYESVariable:
        """
        Create a variable that is identical to the input.
        """
        return DRYESVariable(inputs = {input.name: input},
                             grid_file = grid_file,
                             function = lambda x: x,
                             destination = f'{input.destination}/{input.name}_%Y%m%d.tif',
                             name = input.name)