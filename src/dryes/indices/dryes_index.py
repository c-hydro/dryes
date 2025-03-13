from datetime import datetime, timedelta
from typing import Optional, Sequence
import numpy as np
import copy

from abc import ABC, ABCMeta

from d3tools.timestepping import TimeRange, TimeStep, find_unit_of_time, unit_is_multiple
from d3tools.cases import CaseManager

class MetaDRYESIndex(ABCMeta):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if not hasattr(cls, 'subclasses'):
            cls.subclasses = {}
        elif 'index_name' in attrs:
            cls.subclasses[attrs['index_name']] = cls

    def __new__(cls, name, bases, dct):
        # Create the new class
        new_cls = super().__new__(cls, name, bases, dct)
        
        # Merge default options from parent classes
        default_options = {}
        for base in bases:
            if hasattr(base, 'default_options'):
                default_options.update(base.default_options)
        
        # Update with the subclass's default options
        if 'default_options' in dct:
            default_options.update(dct['default_options'])
        
        new_cls.default_options = default_options
        return new_cls

class DRYESIndex(ABC, metaclass=MetaDRYESIndex):
    index_name = 'dryes_index'

    default_options = {
        'make_parameters': True
        }

    def __init__(self,
                 io_options: dict,
                 index_options: dict = {},
                 run_options: Optional[dict] = None) -> None:

        # set the options (using defaults when not specified)
        self.options = copy.deepcopy(self.default_options)
        self.options.update(index_options)

        # check the data input options first (to see if there is tiles in the input)
        self.io_options = io_options
        self._check_io_data(io_options)

        # set the case tree
        self._set_cases()

        # check the other io options
        self._check_io_parameters(io_options)
        self._check_io_index(io_options)

        if run_options is not None:
            self._set_run_options(run_options)
    
    def _set_cases(self) -> None:
        self.cases = self._get_cases()

    def _get_cases(self) -> dict:

        options = self.options

        # all the options that are not in the self.default_options will be used to set the input
        input_options = {k: v for k, v in options.items() if k not in self.default_options}

        # if the input data has tiles, we need to add the tile name to the input options
        if self._data.has_tiles:
            tile_names = self._data.tile_names
            input_options['tile'] = dict(zip(tile_names, tile_names))

        cases = CaseManager(input_options, 'data')

        # all the other options are divided based on the time in the computation when they are needed
        # this is based on self.option_cases (this only works because dict are now ordered (since python 3.7))
        # if an options isn't there it will use the default value
        for layer, opts in self.option_cases.items():
            these_options = {k:options.get(k, self.default_options.get(k)) for k in opts}
            cases.add_layer(these_options, layer)

        return cases

    def _check_io_data(self, io_options: dict, update_existing = False) -> None:

        if not hasattr(self, '_data') or update_existing:
            self._data = io_options.get('data', None)
            if self._data is None: raise ValueError('No input data specified.')

        self.output_template = self._data._template

    def _check_io_parameters(self, io_options: dict, update_existing = False) -> None:
        if not hasattr(self, '_parameters') or update_existing:
            # check that we have output specifications for all the parameters in self.parameters
            self._parameters = {}
            for par in self.parameters:
                if par not in io_options: raise ValueError(f'No source/destination specified for parameter {par}.')
                self._parameters[par] = io_options[par]
                self._parameters[par]._template = self.output_template

    def _check_io_index(self, io_options: dict, update_existing = False) -> None:
        if not hasattr(self, '_index') or update_existing:
            # check that we have an output specification for the index
            if 'index' not in io_options: raise ValueError('No output path specified for the index.')
            self._index = io_options['index']
            self._index._template = self.output_template

    def _set_run_options(self, run_options: dict) -> None:
        #TODO: check if we can improve/remove some parts of this
        if 'history_start' in run_options and 'history_end' in run_options:
            self.reference = TimeRange(run_options['history_start'], run_options['history_end'])
        else:
            self.reference = None
        
        if 'frequency' in run_options:
            self.freq = find_unit_of_time(run_options['frequency'])
        elif 'timesteps_per_year' in run_options:
            self.freq = find_unit_of_time(timesteps_per_year = run_options['timesteps_per_year'])

    # CLASS METHODS FOR FACTORY
    @classmethod
    def from_options(cls, index_options: dict, io_options: dict, run_options: dict) -> 'DRYESIndex':
        index_name = index_options.pop('index_name', None) or index_options.pop('index', None)
        index_name = cls.get_index_name(index_name)
        Subclass: 'DRYESIndex' = cls.get_subclass(index_name)
        return Subclass(io_options, index_options, run_options)

    @classmethod
    def get_subclass(cls, index_name: str):
        index_name = cls.get_index_name(index_name)
        Subclass: 'DRYESIndex'|None = cls.subclasses.get(index_name)
        if Subclass is None:
            raise ValueError(f"Invalid index: {index_name}")
        return Subclass
    
    @classmethod
    def get_index_name(cls, index_name: Optional[str] = None):
        if index_name is not None:
            return index_name
        elif hasattr(cls, 'index_name'):
            return cls.index_name

    # THESE ARE THE METHODS THAT ARE CALLED IN PRACTICE BY THE USER
    def compute(self, current:   Sequence[datetime]|TimeRange|None = None,
                      reference: Sequence[datetime]|TimeRange|None = None,
                      frequency: int|str|None = None,
                      make_parameters: bool|None = None) -> None:

        # set the make_parameters flag
        if make_parameters is not None:
            self.options['make_parameters'] = make_parameters

        # set current and reference periods
        current   = TimeRange.from_any(current, 'current')
        if current is None and not self.options['make_parameters']:
                raise ValueError('No current period specified.')

        if reference is not None:
            reference = TimeRange.from_any(reference, 'reference')
        elif hasattr(self, 'reference'):
            reference = self.reference
        else:
            raise ValueError('No reference period specified.')

        # set timesteps per year
        if frequency is None:
            if hasattr(self, 'freq'):
                frequency = self.freq

        # calculate the parameters
        if self.options['make_parameters']:
            self._make_parameters(reference, frequency)

        # calculate the index
        if current is not None:
            self._make_index(current, reference, frequency)
    
    def compute_parameters(self,
                           reference: Sequence[datetime]|TimeRange|None = None,
                           frequency: int|str|None = None) -> None:
        
        self.compute(current = None, reference = reference, frequency = frequency, make_parameters = True)

    def get_last_ts(self, inputs = False, **kwargs) -> TimeStep:
        index_cases = self.cases[-1]
        last_ts_index = None
        for case in index_cases.values():
            now = kwargs.pop('now', None) if last_ts_index is None else last_ts_index.end + timedelta(days = 1)
            index = self._index.get_last_ts(now = now, **case.tags, **kwargs)
            if index is not None:
                last_ts_index = index if last_ts_index is None else min(index, last_ts_index)
            else:
                last_ts_index = None
                break
        
        if not inputs:
            return last_ts_index
        
        other = {}
        data_cases = self.cases['data']
        for name, ds in {k:v for k,v in self.io_options.items() if k not in self.parameters and k != "index"}.items():
            last_ts_data = None
            for case in data_cases.values():
                now = kwargs.pop('now', None) if last_ts_data is None else last_ts_data.end + timedelta(days = 1)
                data = ds.get_last_ts(now = now, **case.tags, **kwargs)
                if data is not None:
                    last_ts_data = data if last_ts_data is None else min(data, last_ts_data)
                else:
                    last_ts_data = None
                    break
            other[name] = last_ts_data

        return last_ts_index, other

    # THESE ARE METHODS THAT HANDLE THE CASES, THE DATA, THE PARAMETERS AND THE OUTPUT
    def _make_parameters(self, history: TimeRange, frequency: str|None) -> None:

        for data_case_id, data_case in self.cases['data'].items():

            data_ts_unit = self._data.estimate_timestep(**data_case.options).unit
            if frequency is not None:
                if not unit_is_multiple(frequency, data_ts_unit):
                    raise ValueError(f'The data timestep unit ({data_ts_unit}) is not a multiple of the frequency requeested ({frequency}).')
            else:
                frequency = data_ts_unit
        
            # get the timesteps for which we need to calculate the parameters
            timesteps:list[TimeStep] = TimeRange('1900-01-01', '1900-12-31').get_timesteps(frequency)

            # get the relevant case layers for the parameters
            parameter_layers = [l for l in self.cases._lyrmap if l.startswith('parameters')]

            # loop through all the timesteps
            for time in timesteps:
                # get the data for the relevant timesteps to calculate the parameters
                data_times = time.get_history_timesteps(history)
                all_data_np = []
                for t in data_times:
                    this_data_np = self.get_data(t, data_case)
                    if this_data_np is None:
                        continue ##TODO: ADD A WARNING OR SOMETHING
                    all_data_np.append(this_data_np)
                
                all_data_stacked = np.stack(all_data_np, axis = 0).squeeze()

                # loop through all parameter layers for this data case
                step_input = [None] * len(parameter_layers)
                step_input[0] = all_data_stacked
                for par_case, case_layer in self.cases.iterate_subtree(data_case_id, len(parameter_layers)):
                    # in this case the number of the case layer returned matches the step of the parameter calculation
                    # (because data = 0 and the first parameter = 1) - this will not be true for the index calculation
                    this_input = copy.deepcopy(step_input[case_layer-1])
                    these_parameters = self.calc_parameters(this_input, par_case.options, step = case_layer)

                    if case_layer == len(parameter_layers):
                        for parname, par in these_parameters.items():
                            metadata = par_case.options.copy()
                            metadata.update({'reference': f'{history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}'})
                            tags = par_case.tags
                            self._parameters[parname].write_data(par, time = time, time_format = '%d/%m', metadata = metadata, **tags)
                    else:
                        step_input[case_layer] = these_parameters

    def _make_index(self, current: TimeRange, reference: TimeRange, frequency: str) -> None:

        for data_case_id, data_case in self.cases['data'].items():

            data_ts_unit = self._data.estimate_timestep(**data_case.options).unit
            if frequency is not None:
                if not unit_is_multiple(frequency, data_ts_unit):
                    raise ValueError(f'The data timestep unit ({data_ts_unit}) is not a multiple of the frequency requeested ({frequency}).')
            else:
                frequency = data_ts_unit

            # get the timesteps for which we need to calculate the index
            timesteps:list[TimeStep] = current.get_timesteps(frequency)

            # get the cases for the last layer of the parameters
            parameter_layers = [l for l in self.cases._lyrmap if l.startswith('parameters')]
            this_data_par_cases = {id:case for id,case in self.cases[parameter_layers[-1]].items() if id.startswith(data_case_id)}

            # get the relevant case layers for the index
            index_layers = [l for l in self.cases._lyrmap if l.startswith('index')]

            # loop through all the timesteps
            for time in timesteps:
               
                # get the data for the relevant timesteps to calculate the parameters
                data_np = self.get_data(time, data_case)
                if data_np is None:
                    continue ##TODO: ADD A WARNING OR SOMETHING
                
                # loop through all parameter layers for this data case
                for par_case_id, par_case in this_data_par_cases.items():
                    # get the parameters for this case
                    parameters_np = self.get_parameters(time, par_case)


                    # loop through all index layers for this parameter case
                    step_input = [None] * len(index_layers)
                    step_input[0] = data_np
                    for idx_case, case_layer in self.cases.iterate_subtree(par_case_id, len(index_layers)):
                        # calculate the step n (which depends on how many parameter layers there were)
                        step_n = case_layer - len(parameter_layers)

                        this_input = copy.deepcopy(step_input[step_n-1])
                        this_index = self.calc_index(this_input, parameters_np, idx_case.options, step = step_n)

                        if step_n == len(index_layers):
                            metadata = idx_case.options.copy()
                            metadata.update({'reference': f'{reference.start:%d/%m/%Y}-{reference.end:%d/%m/%Y}'})
                            tags = idx_case.tags

                            if isinstance(this_index, list):
                                for index, other_tags in this_index:
                                    self._index.write_data(index, time = time, metadata = metadata, **tags, **other_tags)
                            else:
                                self._index.write_data(this_index, time = time, metadata = metadata, **tags)
                        else:
                            step_input[step_n] = this_index
    
    # make things more flexible, but creating methods to get the data and parameters
    def get_data(self, time: datetime, case) -> np.ndarray:
        if not self._data.check_data(time, **case.options):
            return None

        data_xr = self._data.get_data(time, **case.options)
        data_np = data_xr.values.squeeze()
        return data_np
    
    def get_parameters(self, time: datetime, case) -> dict[str, np.ndarray]:
        parameters_xr = {parname: self._parameters[parname].get_data(time, **case.tags) for parname in self.parameters}
        parameters_np = {parname: par.values.squeeze() for parname, par in parameters_xr.items()}
        return parameters_np

    # THESE ARE THE METHODS THAT NEED TO BE IMPLEMENTED BY THE SUBCLASSES
    def calc_parameters(self,
                        data: list[np.ndarray]|dict[str, np.ndarray], 
                        options: dict, step = 1, **kwargs) -> dict[str, np.ndarray]:
        """

        """
        raise NotImplementedError
    
    def calc_index(self,
                   data: np.ndarray, parameters: dict[str, np.ndarray],
                   options: dict, step = 1, **kwargs) -> np.ndarray:
        """

        """
        raise NotImplementedError
    