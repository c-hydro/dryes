from datetime import datetime
from typing import Sequence, Optional
from copy import deepcopy
import xarray as xr
import numpy as np

from d3tools import timestepping as ts

from .dryes_index import DRYESIndex
from ..index_combination import COMBINATION_ALGORITHMS

class DRYESCombinedIndicator(DRYESIndex):
    index_name = 'indicator'
    default_options = {
        'algorithm': 'cdi_jrc',
        'args': {}
    }

    # we need to have these defined for the code in DRYESIndex to work, but they are not used for this class
    option_cases = {}
    parameters = []

    def __init__(self,
                 io_options: dict,
                 index_options: dict = {},
                 run_options: Optional[dict] = None) -> None:

        # set the algorithm 
        algorithm = index_options.pop('algorithm', self.default_options['algorithm'])
        self.algorithm = COMBINATION_ALGORITHMS.get(algorithm)
        self.args      = index_options.pop('args', self.default_options['args'])

        # set the various keys that the algorithm needs
        self.input_keys    = self.algorithm.input
        self.output_keys   = self.algorithm.output
        self.previous_keys = self.algorithm.previous if hasattr(self.algorithm, 'previous') else []
        self.static_keys   = self.algorithm.static   if hasattr(self.algorithm, 'static')   else []
        
        # run the super init for everything else
        super().__init__(io_options, index_options, run_options)

    def _check_io_data(self, io_options: dict, update_existing = False) -> None:

        # ensure that all keys that are needed are present
        for key in self.input_keys + self.previous_keys + self.static_keys:
            if key not in io_options:
                raise ValueError(f'Input data {key} not found in io_options.')

        # raw_inputs are only the input keys, the previous keys are most likely the output of the algorithm
        # this is important to note because self.raw_inputs is used to find the last available timestep to run the algorithm
        self._raw_inputs       = {k:io_options[k] for k in self.input_keys}
        self._raw_previnputs   = {k:io_options[k] for k in self.previous_keys}
        self._raw_staticinputs = {k:io_options[k] for k in self.static_keys}

        # ensure that we have the same template for all datasets
        self.output_template = self._raw_inputs[self.input_keys[0]]._template
        for i in [self._raw_inputs, self._raw_previnputs, self._raw_staticinputs]:
            for k in i:
                i[k]._template = self.output_template

        self._data = self._raw_inputs[self.input_keys[0]]

    def _check_io_index(self, io_options: dict, update_existing = False) -> None:
    
        # check that all the output keys are present
        self._raw_outputs = {}
        for i, key in enumerate(self.output_keys):
            if key in io_options:
                self._raw_outputs[key] = io_options[key]
            else:
                if i == 0: # the first key might be substituted by "index" and it should still work
                    if 'index' in io_options:
                        self._raw_outputs[key] = io_options['index']
                    else:
                        raise ValueError(f'Output data "{key}" or "index" not found in io_options.')
                else:  
                    raise ValueError(f'Output data "{key}" not found in io_options.')

        self._index = self._raw_outputs[self.output_keys[0]]

        # # make the self._index attribute a MemoryDataset, with the {outkey} in the pattern
        # # then we will handle how to write the data in the save_output_data method later
        # from d3tools.data.memory_dataset import MemoryDataset
        # key_pattern = self._raw_outputs[self.output_keys[0]].key_pattern.replace(self.output_keys[0], '{outkey}')
        # self._index = MemoryDataset(key_pattern)
        # if self._index.has_tiles: self._index.tile_names = self._raw_outputs[self.output_keys[0]].tile_names
        # self._index._template = self.output_template

    def _set_run_options(self, run_options: dict) -> None:

        # reference period is not used in this class, but it is set to None
        if 'history_start' in run_options or 'history_end' in run_options:
            run_options.pop('history_start', None)
            run_options.pop('history_end', None)
        
        super()._set_run_options(run_options)
        self.reference = None

    # THESE ARE THE METHODS THAT ARE CALLED IN PRACTICE BY THE USER
    def compute(self, current:   Sequence[datetime]|ts.TimeRange|None = None,
                      reference: Sequence[datetime]|ts.TimeRange|None = None,
                      frequency: int|str|None = None,
                      make_parameters: bool|None = None) -> None:
        
        # the reference period is not used and there is no parameters here
        super().compute(current, None, frequency, False)

    def _make_index(self, current: ts.TimeRange, reference: ts.TimeRange, frequency: str) -> None:

        for data_case in self.cases['data'].values():

            data_ts_unit = self._data.estimate_timestep(**data_case.options).unit
            if frequency is not None:
                if not ts.unit_is_multiple(frequency, data_ts_unit):
                    raise ValueError(f'The data timestep unit ({data_ts_unit}) is not a multiple of the frequency requeested ({frequency}).')
            else:
                frequency = data_ts_unit

            # get the timesteps for which we need to calculate the index
            timesteps:list[ts.TimeStep] = current.get_timesteps(frequency)

            # get the static data for this case
            static_data = self.get_static_data(data_case)

            # loop through all the timesteps
            for time in timesteps:
               
                # get the data for the relevant timesteps to calculate the parameters
                input_data = self.get_data(time, data_case)
                if input_data is None:
                    continue ##TODO: ADD A WARNING OR SOMETHING
                
                previous_data = self.get_previous_data(time, data_case)

                args = {'input_data': input_data, 'previous_data': previous_data, 'static_data': static_data}
                index_outputs = self.algorithm(**args, **self.args)

                for i, k in enumerate(self.output_keys):
                    self._raw_outputs[k].write_data(index_outputs[k], time = time, metadata = data_case.options, **data_case.tags)

    def get_data(self, time: ts.TimeStep, case):
        # get the data for the relevant timesteps to calculate the parameters
        input_data = {}
        for k,d in self._raw_inputs.items():
            if d.check_data(time, **case.tags):
                input_data[k] = d.get_data(time, **case.tags)
            else:
                input_data[k] = None

        return input_data

    def get_previous_data(self, time: ts.TimeStep, case):
        # get the static data for this case
        previous_data = {}
        for k,d in self._raw_previnputs.items():
            if d.check_data(time - 1, **case.tags):
                previous_data[k] = d.get_data(time - 1,**case.tags)
            else:
                previous_data[k] = None

        return previous_data 

    def get_static_data(self, case):
        # get the static data for this case
        static_data = {}
        for k,d in self._raw_staticinputs.items():
            if d.check_data(**case.tags):
                static_data[k] = d.get_data(**case.tags)
            else:
                static_data[k] = None

        return static_data  
