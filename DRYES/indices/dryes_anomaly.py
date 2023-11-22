from datetime import datetime
from typing import Callable

from .dryes_index import DRYESIndex
from ..variables.dryes_variable import DRYESVariable
from ..time_aggregation import TimeAggregation
from..lib.time import TimeRange

class DRYESAnomaly(DRYESIndex):
    def __init__(self, input_variable: DRYESVariable,
                 time_aggregation: TimeAggregation,
                 output_paths: dict,
                 log_file: str = 'DRYES_log.txt') -> None:

        super().__init__(input_variable, time_aggregation, output_paths, log_file)

