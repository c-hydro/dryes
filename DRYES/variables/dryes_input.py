from __future__ import annotations

import os
from typing import Optional

from ..data_sources import DRYESDataSource
from ..lib.log import log
from ..lib.io import save_dataset_to_geotiff, check_data
from ..lib.time import TimeRange
from ..lib.space import Grid

class DRYESInput():
    def __init__(self, data_source: DRYESDataSource, variable: str, type: str, destination: str) -> None:
        self.data_source = data_source
        self.start = data_source.get_start()
        self.isstatic = type == 'static'
        self.destination = destination
        self.addn = '_%Y%m%d' if not self.isstatic else ''
        self.path = os.path.join(self.destination, variable+self.addn+'.tif')
        self.name = variable

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
            timesteps_to_download = [time for time in timesteps_to_download if not check_data(self.path, time)]
            num_timesteps = len(timesteps_to_download)
            if num_timesteps == 0:
                log(f'All timesteps already available locally.')
                return
            
            log(f'Found {num_timesteps} timesteps not already available locally.')

            # download each remaining timestep
            for time in timesteps_to_download:
                log(f'Downloading {variable_name} for {time:%Y-%m-%d}.')
                data = source.get_data(grid, time)
                output_path = time.strftime(self.destination)
                addn = time.strftime(self.addn)

                os.makedirs(output_path, exist_ok=True)
                n_saved = save_dataset_to_geotiff(data, output_path, addn)
        
                log(f'Saved {n_saved} files to {output_path}')