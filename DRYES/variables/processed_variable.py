from __future__ import annotations

import xarray as xr

from typing import Callable, Optional

from ..lib.log import log
from ..lib.io import save_dataarray_to_geotiff, check_data, check_data_range, get_data
from ..lib.time import TimeRange
from ..lib.space import Grid

from .dryes_variable import DRYESVariable

class ProcessedVariable(DRYESVariable):
    def __init__(self, inputs: dict[str:DRYESVariable],\
                       #grid_file: str,
                       function: Callable[..., xr.DataArray],
                       destination: str,
                       name: Optional[str]=None) -> None:
        
        self.inputs = inputs
        self.start = max([v.start for v in inputs.values() if not v.isstatic])
        #self.grid = Grid(grid_file)

        self.function = function
        
        if name is None: name = 'variable'
        self.name = name
        self.path = destination

    def gather_inputs(self, grid: Grid, time_range: TimeRange) -> None:
        """
        Gathers all the data from the remote source in the TimeRange,
        also checks that the data is not available yet before gathering it
        """

        log(' Checking source data:')
        for v in self.inputs.values():
            v.make(grid, time_range)

    def compute(self, grid, time_range: TimeRange):
        """
        Computes the data from the other inputs in the TimeRange,
        also checks that the data is not available yet before computing it
        """
        
        log(f' Processing source data into {self.name}:')
        variable_paths = [v.path for v in self.inputs.values()]

        timesteps_to_compute_per_var = [set(check_data_range(paths, time_range)) for paths in variable_paths]
        intersection = set.intersection(*timesteps_to_compute_per_var)
        timesteps_to_compute = list(intersection)
        timesteps_to_compute.sort() # sort the timesteps in chronological order <- going through the set messes up the order
        tot_timesteps = len(timesteps_to_compute)

        # filter out the timesteps that are already computed
        timesteps_to_compute = [time for time in timesteps_to_compute if not check_data(self.path, time)]
        num_timesteps = len(timesteps_to_compute)
        log(f' - {tot_timesteps - num_timesteps}/{tot_timesteps} timesteps available locally.')
        if num_timesteps == 0:
            return
        
        log(f' - Processing {num_timesteps} timesteps.')
        # get the static inputs, these are the same for each timestep
        static_data = {k:get_data(v.path) for k,v in self.inputs.items() if v.isstatic}

        # compute each remaining timestep
        data_template = grid.template
        for time in timesteps_to_compute:
            log(f'   - Processing {time:%Y-%m-%d}...')
            dynamic_data = {k:get_data(v.path, time) for k,v in self.inputs.items() if not v.isstatic}
            data = self.function(**static_data, **dynamic_data)
            output_file = time.strftime(self.path)
            output = data_template.copy(data = data)
            metadata = {'name' : self.name,
                        'type' : 'DRYES data',
                        'aggregation': 'none',
                        'time' : time.strftime('%Y-%m-%d')}
            saved = save_dataarray_to_geotiff(output, output_file, metadata)
            if saved:
                log(f'   - Saved to {output_file}')

    def make(self, grid: Grid, time_range: TimeRange = None):
        """
        Gathers the data from the remote source in the TimeRange
        and preprocesses it using the function
        """

        self.gather_inputs(grid, time_range)
        self.compute(grid, time_range)

    # @staticmethod
    # def identical(input: DRYESInput, grid_file: str) -> DRYESVariable:
    #     """
    #     Create a variable that is identical to the input.
    #     """
    #     return DRYESVariable(inputs = {input.name: input},
    #                          grid_file = grid_file,
    #                          function = lambda x: x,
    #                          destination = f'{input.destination}/{input.name}_%Y%m%d.tif',
    #                          name = input.name)