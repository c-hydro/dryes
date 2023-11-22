import xarray as xr
from typing import Optional, Callable
from datetime import datetime

from ..variables import DRYESVariable
from ..lib.log import log
from ..lib.time import TimeRange, create_timesteps
from ..lib.io import check_data_range, save_dataarray_to_geotiff

AggFunction = Callable[[DRYESVariable, datetime,], xr.Dataset]
PostAggFunction = Callable[[list[xr.DataArray], DRYESVariable], list[xr.DataArray]]

class TimeAggregation:
    def __init__(self, timesteps_per_year: int, aggregation_function: Optional[AggFunction] = None) -> None:
        self.timesteps_per_year = timesteps_per_year
        if aggregation_function is not None:
            self.aggfun = {'Agg' : aggregation_function}
        else:
            self.aggfun = {}
        self.postaggfun = {}

    def add_aggregation(self, name:str,\
                        function: AggFunction,\
                        post_function: Optional[PostAggFunction] = None) -> 'TimeAggregation':
        """
        Adds an aggregation (and possibly a post_aggregation) function to the TimeAggregation object.
        Aggregation functions operate on individual timesteps, post_aggregation functions on all timesteps at once:
            they are useful for example for exponential smoothing.
        """
        self.aggfun[name] = function
        if post_function is not None:
            self.postaggfun[name] = post_function
        return self

    def apply(self, input: DRYESVariable, time_range: TimeRange, destination: str) -> xr.Dataset:
 
        var = input.name
        
        all_timesteps = create_timesteps(time_range.start, time_range.end, self.timesteps_per_year)

        agg_names = list(self.aggfun.keys())
        agg_paths = {agg_name:destination.format(var = var, time_agg = agg_name) for agg_name in agg_names}
        available_timesteps = {name:list(check_data_range(path, time_range)) for name,path in agg_paths.items()}

        timesteps_to_compute = {}
        for agg_name in agg_names:
            # if there is no post aggregation function, we don't care for the order of the timesteps
            if agg_name not in self.postaggfun.keys():
                timesteps_to_compute[agg_name] = [time for time in all_timesteps if time not in available_timesteps[agg_name]]
            # if there is a post aggregation function, we need to compute the timesteps in order
            else:
                timesteps_to_compute[agg_name] = []
                i = 0
                while(all_timesteps[i] not in available_timesteps[agg_name]) and i < len(all_timesteps):
                    timesteps_to_compute[agg_name].append(all_timesteps[i])
                    i += 1

        timesteps_to_iterate = set.union(*[set(timesteps_to_compute[agg_name]) for agg_name in agg_names])
        timesteps_to_iterate = list(timesteps_to_iterate)
        timesteps_to_iterate.sort()
        
        agg_data = {n:[] for n in agg_names}
        for time in timesteps_to_iterate:
            for agg_name in agg_names:
                if time in timesteps_to_compute[agg_name]:
                    agg_data[agg_name].append(self.aggfun[agg_name](input, time))
        
        for agg_name in agg_names:
            if agg_name in self.postaggfun.keys():
                agg_data[agg_name] = self.postaggfun[agg_name](agg_data[agg_name], input)

            for i, data in enumerate(agg_data[agg_name]):
                this_time = timesteps_to_compute[agg_name][i]
                path_out = this_time.strftime(agg_paths[agg_name])
                save_dataarray_to_geotiff(data, path_out)

        return agg_paths