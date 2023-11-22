from datetime import datetime
from typing import Callable

from .dryes_index import DRYESIndex
from ..variables.dryes_variable import DRYESVariable
from ..time_aggregation import TimeAggregation

from ..lib.time import TimeRange
from ..lib.log import log

class DRYESAnomaly(DRYESIndex):
    def __init__(self, input_variable: DRYESVariable,
                 time_aggregation: TimeAggregation,
                 options: dict,
                 output_paths: dict,
                 log_file: str = 'DRYES_log.txt') -> None:

        super().__init__(input_variable, time_aggregation, options, output_paths, log_file)

    def check_options(self, options: dict) -> dict:
        if not 'type' in options:
            log('No anomaly type specified, using default type: empirical z-score.')
            options['type'] = 'empiricalzscore'
        
        opts = {}
        opts['type'] = options['type']
        return opts