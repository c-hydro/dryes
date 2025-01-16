import xarray as xr
from typing import Optional, Callable
from datetime import datetime

from d3tools.data import Dataset
from d3tools.timestepping.timestep import TimeStep

AggFunction = Callable[[Dataset, datetime,], xr.Dataset]
PostAggFunction = Callable[[list[xr.DataArray], Dataset], list[xr.DataArray]]

global TIMEAGG_FUNCTIONS
TIMEAGG_FUNCTIONS = {'agg': {}, 'postagg': {}}

def as_timagg_function(type = 'agg'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        TIMEAGG_FUNCTIONS[type][func.__name__] = wrapper
        return wrapper
    return decorator

from .aggregation_functions import *

class TimeAggregation:
    def __init__(self, #timesteps_per_year: int,
                 aggregation_function: Optional[AggFunction] = None) -> None:
        #self.timesteps_per_year = timesteps_per_year
        if aggregation_function is not None:
            self.aggfun = {'Agg' : aggregation_function}
        else:
            self.aggfun = {}
        self.postaggfun = {}
        self.tags = {}

    def add_aggregation(self, name:str,
                        function: AggFunction|dict,
                        post_function: Optional[PostAggFunction|dict] = None) -> 'TimeAggregation':
        """
        Adds an aggregation (and possibly a post_aggregation) function to the TimeAggregation object.
        Aggregation functions operate on individual timesteps, post_aggregation functions on all timesteps at once:
            they are useful for example for exponential smoothing.
        """
        if isinstance(function, dict):
            function = self.make_aggfn(function)
        self.aggfun[name] = function
        if post_function is not None:
            if isinstance(post_function, dict):
                post_function = self.make_aggfn(post_function, type = 'postagg')
            self.postaggfun[name] = post_function
        return self
    
    @staticmethod
    def make_aggfn(fundict: dict, type: str = 'agg') -> Callable:
        fun_name = fundict.pop('fun')
        fun = TIMEAGG_FUNCTIONS[type][fun_name](**fundict)
        return fun

    def aggregate_data(self, timesteps_to_compute: dict[str:list[TimeStep]], variable_in: Dataset, variable_out: Dataset) -> None:
                       
        """
        Aggregates the input data in variable_in and writes the output in variable_out.
        The aggregation is performed according to the aggregation functions in self.aggfun and self.postaggfun.

        The timesteps to compute are passed as a dictionary of lists of TimeStep objects, where the keys are the names of the aggregations to perform.
        """

        # put all the timesteps in the same list, this is what we have to iterate over
        timesteps_to_iterate = []
        for agg_ts in timesteps_to_compute.values():
            timesteps_to_iterate += [ts for ts in agg_ts if ts not in timesteps_to_iterate]
    
        timesteps_to_iterate.sort()

        if len(timesteps_to_iterate) == 0:
            return

        for this_ts in timesteps_to_iterate:
            for agg_name in timesteps_to_compute.keys():
                agg_fn = self.aggfun.get(agg_name)
                if this_ts in timesteps_to_compute[agg_name]:
                    time_varin = this_ts.end if hasattr(this_ts, 'end') else this_ts
                    agg_data, agg_info = agg_fn(variable_in, time_varin)
                    if agg_data is None:
                        agg_data = variable_out.build_templatearray(variable_in.get_template_dict())
                    if agg_name not in self.postaggfun.keys():
                        variable_out.write_data(agg_data, time = this_ts, metadata = agg_info, agg_fn = agg_name)
                    else:
                        time_varout = variable_out.get_time_signature(this_ts)
                        postagg_data, postagg_info = self.postaggfun[agg_name](agg_data, variable_out, time_varout)
                        for key in postagg_info:
                            if key in agg_info:
                                agg_info[key] += f'; {postagg_info[key]}'
                            else:
                                agg_info[key] = postagg_info[key]
                        variable_out.write_data(postagg_data, time = this_ts, metadata = agg_info, agg_fn = agg_name)