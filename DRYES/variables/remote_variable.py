from __future__ import annotations

import os
from typing import Optional

from ..data_sources import DRYESDataSource
from ..lib.log import log
from ..lib.io import save_dataset_to_geotiff, check_data
from ..lib.time import TimeRange
from ..lib.space import Grid

from .dryes_variable import DRYESVariable

class RemoteVariable(DRYESVariable):
    def __init__(self, data_source: DRYESDataSource, variable: str, type: str, destination: str) -> None:
        self.data_source = data_source
        self.start = data_source.get_start()
        self.isstatic = type == 'static'
        self.destination = destination
        self.addn = '_%Y%m%d' if not self.isstatic else ''
        self.path = os.path.join(self.destination, variable+self.addn+'.tif')
        self.name = variable

    def make(self, grid: Grid, time_range: Optional[TimeRange] = None) -> None:
        """
        Gathers all the data from the remote source in the TimeRange,
        also checks that the data is not available yet before gathering it
        """

        source = self.data_source
        source_name = source.__class__.__name__
        variable_name = source.variable

        if time_range is None and not self.isstatic:
            raise ValueError(f'TimeRange must be specified for dynamic data: {variable_name}.')
        
        if self.isstatic:
            # check if the data is already available locally
            if check_data(self.path):
                log(f' - {variable_name} (static) available locally.')
                return

            # download the data
            log(f' - {variable_name} (static): gathering using {source_name}.')
            data = source.get_data(grid)
            output_path = self.destination
            n_saved = save_dataset_to_geotiff(data, output_path)

            log(f' - {variable_name} (static): saved {n_saved} files to {output_path}')
        
        else:
            # get all the timesteps that the source is available for in the timerange
            timesteps_to_download = list(source.get_times(time_range.start, time_range.end))
            tot_timesteps = len(timesteps_to_download)

            # filter out the timesteps that are already available locally
            timesteps_to_download = [time for time in timesteps_to_download if not check_data(self.path, time)]
            num_timesteps = len(timesteps_to_download)
            log(f' - {variable_name}: {tot_timesteps - num_timesteps}/{tot_timesteps} timesteps available locally.')
            if num_timesteps == 0:
                return
            
            # download each remaining timestep
            log(f' - {variable_name}: gathering {num_timesteps} timesteps using {source_name}.')
            for time in timesteps_to_download:
                log(f'   - Gathering {time:%Y-%m-%d}...')
                data = source.get_data(grid, time)
                output_path = time.strftime(self.destination)
                addn = time.strftime(self.addn)

                os.makedirs(output_path, exist_ok=True)
                n_saved = save_dataset_to_geotiff(data, output_path, addn)
        
                log(f'   - Saved {n_saved} files to {output_path}')