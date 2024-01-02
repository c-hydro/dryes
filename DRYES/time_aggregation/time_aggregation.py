import xarray as xr
from typing import Optional, Callable
from datetime import datetime

from ..io import IOHandler

AggFunction = Callable[[IOHandler, datetime,], xr.Dataset]
PostAggFunction = Callable[[list[xr.DataArray], IOHandler], list[xr.DataArray]]

class TimeAggregation:
    def __init__(self, #timesteps_per_year: int,
                 aggregation_function: Optional[AggFunction] = None) -> None:
        #self.timesteps_per_year = timesteps_per_year
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
