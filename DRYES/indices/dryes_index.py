from datetime import datetime
from typing import Callable
import itertools
import copy

from ..variables.dryes_variable import DRYESVariable
from ..time_aggregation import TimeAggregation

from ..lib.log import setup_logging
from ..lib.time import TimeRange, create_timesteps
from ..lib.parse import substitute_values

class DRYESIndex:
    def __init__(self, input_variable: DRYESVariable,
                 time_aggregation: TimeAggregation,
                 options: dict,
                 output_paths: dict,
                 log_file: str = 'DRYES_log.txt') -> None:
        
        setup_logging(log_file)

        self.input_variable = input_variable
        self.time_aggregation = time_aggregation
        self.timesteps_per_year = time_aggregation.timesteps_per_year

        options = self.check_options(options)
        self.options = options

        self.get_cases()
        self.output_paths = substitute_values(output_paths, output_paths, rec = False)
    
    def get_cases(self) -> list:

        options = self.options

        # get the options that need to be permutated and the ones that are fixed
        fixed_options = {k: v for k, v in options.items() if not isinstance(v, dict)}
        to_permutate = {k: list(v.keys()) for k, v in options.items() if isinstance(v, dict)}
        values_to_permutate = [v for v in to_permutate.values()]
        keys = list(to_permutate.keys())

        permutations = [dict(zip(keys, p)) for p in itertools.product(*values_to_permutate)]
        identifiers = copy.deepcopy(permutations)
        for permutation in permutations:
            permutation.update(fixed_options)

        cases_opts = []
        for permutation in permutations:
            this_case_opts = {}
            for k, v in permutation.items():
                # if this is one of the options that we permutated
                if isinstance(options[k], dict):
                    this_case_opts[k] = options[k][v]
                # if not, this is fixed
                else:
                    this_case_opts[k] = v
                    permutation[k] = ""
            cases_opts.append(this_case_opts)

        cases = []
        for case, permutation, i in zip(cases_opts, permutations, range(len(identifiers))):
            this_case = dict()
            this_case['id'] = i
            this_case['tags'] = {'options.' + pk:pv for pk,pv in permutation.items()}
            this_case['options'] = case
            cases.append(this_case)

        # cases = [{"options": c, \
        #           "tags"   : {'options.' + pk:pv for pk,pv in p.items()}, \
        #           "id"     : i} for c,p,i in zip(cases_opts, permutations, range(len(identifiers)))]

        self.cases = cases

    def check_options(self, options: dict) -> dict:
        return options

    def make_data_range(self, current: TimeRange, reference: TimeRange|Callable[[datetime], TimeRange]):

        # all of this will get the range for the data that is needed
        current_timesteps = create_timesteps(current.start, current.end, self.timesteps_per_year)
        reference_start = set(reference(time).start for time in current_timesteps)

        time_start = min(reference_start)
        time_end   = max(current_timesteps)

        return TimeRange(time_start, time_end)

    def compute(self, current: TimeRange, reference: TimeRange|Callable[[datetime], TimeRange]) -> None:
        
        if isinstance(reference, TimeRange):
            reference = lambda time: reference

        # get the range for the data that is needed
        data_range = self.make_data_range(current, reference)

        # get the data -> this will both gather and compute the data (checking if it is already available)
        self.input_variable.make(data_range)

        breakpoint()

        pass