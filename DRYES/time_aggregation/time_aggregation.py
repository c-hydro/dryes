import xarray as xr
from typing import Optional, Callable
from datetime import datetime

from ..variables import DRYESVariable
from ..lib.time import TimeRange, create_timesteps
from ..lib.log import log

class TimeAggregation:
    def __init__(self, timesteps_per_year: int, aggregation_function: Optional[dict[str:Callable]] = None) -> None:
        self.timesteps_per_year = timesteps_per_year
        if aggregation_function is not None:
            self.aggfun = aggregation_function
        else:
            self.aggfun = {}
        self.postaggfun = {}

    def add_aggregation(self, name:str,\
                        function: Callable[[DRYESVariable, datetime,], xr.Dataset],\
                        post_function: Optional[Callable[[xr.Dataset], xr.Dataset]] = None) -> 'TimeAggregation':
        """
        Adds an aggregation (and possibly a post_aggregation) function to the TimeAggregation object.
        Aggregation functions operate on individual timesteps, post_aggregation functions on all timesteps at once:
            they are useful for example for exponential smoothing.
        """
        self.aggfun[name] = function
        if post_function is not None:
            self.postaggfun[name] = post_function
        return self
    
    def __call__(self, variable: DRYESVariable, time_range: TimeRange) -> xr.Dataset:
        """
        Aggregates the data in a DRYESDataset object over time.
        """
        timesteps = create_timesteps(time_range.start, time_range.end, self.timesteps_per_year)

        log(f'Aggregating {variable.name} over {len(timesteps)} timesteps between {timesteps[0]:%Y-%m-%d} and {timesteps[-1]:%Y-%m-%d}.')
        log(f'{len(self.aggfun)} aggregation(s) found: {", ".join(self.aggfun.keys())}')

        agg_data = {n: [] for n in self.aggfun.keys()}
        for time in timesteps:
            log(f'Starting timestep {time:%Y-%m-%d}.')
            for name, f in self.aggfun.items():
                this_agg_data = f(variable, time)
                if this_agg_data is not None:
                    agg_data[name].append(this_agg_data)
        
        log(f'Completing aggregation...')
        for name, f in self.postaggfun.items():
            agg_data[name] = f(agg_data[name])

        agg_data = {n: xr.concat(d, dim = 'time') for n, d in agg_data.items()}
        varname = variable.name
        for n in agg_data.keys(): agg_data[n].name = ('_').join([varname, n])
        agg_data = xr.merge(agg_data.values())

        log(f'Aggregation complete.')

        return agg_data

