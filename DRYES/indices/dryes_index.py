from datetime import datetime
from typing import Callable

from ..variables.dryes_variable import DRYESVariable
from ..time_aggregation import TimeAggregation
from ..lib.log import setup_logging
from ..lib.time import TimeRange, create_timesteps

class DRYESIndex:
    def __init__(self, input_variable: DRYESVariable,
                 time_aggregation: TimeAggregation,
                 output_paths: dict,
                 log_file: str = 'DRYES_log.txt') -> None:
        
        self.input_variable = input_variable
        self.time_aggregation = time_aggregation
        self.timesteps_per_year = time_aggregation.timesteps_per_year
        self.output_paths = output_paths
        setup_logging(log_file)

    def make_data_range(self, current: TimeRange, reference: TimeRange|Callable[[datetime], TimeRange]):

        # all of this will get the range for the data that is needed
        current_timesteps = create_timesteps(current.start, current.end, self.timesteps_per_year)
        reference_start = set(reference(time).start for time in current_timesteps)

        time_start = min(reference_start)
        time_end   = max(current_timesteps)

        return TimeRange(time_start, time_end)

    def compute(self, current: TimeRange, reference: TimeRange|Callable[[datetime], TimeRange]) -> None:
        
        # get the range for the data that is needed
        data_range = self.make_data_range(current, reference)

        # get the data -> this will both gather and compute the data (checking if it is already available)
        self.input_variable.make(data_range)

        breakpoint()

        pass