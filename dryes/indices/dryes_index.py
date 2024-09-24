from datetime import datetime
from typing import Callable, Iterable, Optional
import numpy as np
from copy import copy, deepcopy
import logging
import os

from abc import ABC, ABCMeta

from ..time_aggregation.time_aggregation import TimeAggregation

from ..utils.parse import options_to_cases
from ..tools.timestepping import TimeRange
from ..tools.timestepping.fixed_num_timestep import FixedNTimeStep
from ..tools.timestepping.timestep import TimeStep
from ..tools.data import Dataset

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
        
        # set the logging
        index_name = self.index_name
        self.log = logging.getLogger(f'{index_name}')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        if 'log' in io_options:
            logpath = io_options['log'].path()
            os.makedirs(os.path.dirname(logpath), exist_ok=True)

            #logging.basicConfig(filename = logpath, level = logging.INFO,
            #                    format = '%(asctime)s - %(name)s - %(message)s', force=True)
    
            handler_file = logging.FileHandler(logpath)
            handler_file.setFormatter(formatter)
            self.log.addHandler(handler_file)
        else:
            handler_console = logging.StreamHandler()
            handler_console.setFormatter(formatter)
            self.log.addHandler(handler_console)

        self._check_index_options(index_options)
        self._get_cases()

        self._check_io_options(io_options)

        if run_options is not None:
            self._set_run_options(run_options)
    
    def _check_index_options(self, options: dict) -> None:
        these_options = self.default_options.copy()
        these_options.update({'post_fn': None})
        for k in these_options.keys():
            if k in options:
                these_options[k] = options[k]
            else:
                self.log.info(f'No option {k} specified, using default value: {these_options[k]}.')

        # all of this is a way to divide the options into three bits:
        # - the time aggregation, which will affect the input data (self.time_aggregation)
        # - the options that will be used to calculate the parameters and the index (self.options)
        # - the post-processing function, which will be applied to the index (self.post_processing)
        options = self._make_time_aggregation(these_options)
        self.post_processing = options['post_fn']
        del options['post_fn']
        self.options = options

    def _make_time_aggregation(self, options: dict) -> dict:
        # if the iscontinuous flag is not set, set it to False
        if not hasattr(self, 'iscontinuous'):
                self.iscontinuous = False

        # deal with the time aggregation options
        if not 'agg_fn' in options:
            self.time_aggregation = TimeAggregation()
            return options
        
        agg_options = options['agg_fn']
        if not isinstance(agg_options, dict):
            agg_options = {'agg': agg_options}
        
        time_agg = TimeAggregation()
        for agg_name, agg_fn in agg_options.items():
            if isinstance(agg_fn, Callable|str):
                time_agg.add_aggregation(agg_name, agg_fn)
            elif isinstance(agg_fn, tuple) or isinstance(agg_fn, list):
                if len(agg_fn) == 1:
                    time_agg.add_aggregation(agg_name, agg_fn[0])
                else:
                    time_agg.add_aggregation(agg_name, agg_fn[0], agg_fn[1])
                    # if we have a post-aggregation function, the calculation needs to be continuous
                    self.iscontinuous = True
            else:
                raise ValueError(f'Aggregation function {agg_name} not recognized.')

        self.time_aggregation = time_agg

        del options['agg_fn']

        return options

    def _get_cases(self) -> None:

        # Similarly to the options, we have three layers of cases to deal with
        time_agg = list(self.time_aggregation.aggfun.keys()) # cases[0]
        options = self.options                               # cases[1]
        post_processing = self.post_processing               # cases[2]

        ## start with the time_aggregation
        agg_cases = []
        # if len(time_agg) == 0:
        #     agg_cases = [None]

        for i, agg_name in enumerate(time_agg):
            this_case = dict()
            this_case['id']   = i
            this_case['name'] = agg_name
            this_case['tags'] = {'agg_fn': agg_name}
            agg_cases.append(this_case)

        ## then add the options
        opt_cases = options_to_cases(options)

        ## finally add the post-processing, if it exists
        post_cases = []
        if post_processing is not None:
            i = 0
            for post_name, post_fn in post_processing.items():
                this_case = dict()
                this_case['id']   = i
                this_case['name'] = post_name
                this_case['tags'] = {'post_fn': post_name}
                this_case['post_fn'] = post_fn
                post_cases.append(this_case)
                i += 1

        ## and combine them
        self.cases = {'agg': agg_cases, 'opt': opt_cases, 'post': post_cases}

    def _check_io_options(self, io_options: dict, update_existing = False) -> None:
        # check that we have all the necessary options
        self._check_io_data(io_options, update_existing)
        self._check_io_parameters(io_options, update_existing)
        self._check_io_index(io_options, update_existing)

    def _check_io_data(self, io_options: dict, update_existing = False) -> None:
        
        # for most indices, we need 'data' (for aggregated input data) and 'data_raw' (for raw input data)
        # if we don't have 'data_raw', we check if the data needs to be aggregated, if not we can use 'data'
        has_agg = len(self.cases['agg']) > 0
        if has_agg:
            # if we have aggregations, we need both 'data' and 'data_raw'
            if 'data_raw' not in io_options or 'data' not in io_options:
                raise ValueError('Both data and data_raw must be specified.')
            # otherwise, we still need at least one of them
        elif 'data' not in io_options and 'data_raw' not in io_options:
            raise ValueError('Either data or data_raw must be specified.')
        elif 'data' not in io_options:
            io_options['data'] = io_options['data_raw']
        elif 'data_raw' not in io_options:
            io_options['data_raw'] = io_options['data']

        if not hasattr(self, '_raw_data') or update_existing:
            self._raw_data = io_options['data_raw']
            self._raw_data._template = {}

        if not hasattr(self, '_data') or update_existing:
            self._data = io_options['data']
            self._data._template = self._raw_data._template

        self.output_template = self._raw_data._template

    def _check_io_parameters(self, io_options: dict, update_existing = False) -> None:
        if not hasattr(self, '_parameters') or update_existing:
            # check that we have output specifications for all the parameters in self.parameters
            self._parameters = {}
            for par in self.parameters:
                if par not in io_options:
                    raise ValueError(f'No output path for parameter {par}.')
                self._parameters[par] = io_options[par]
                self._parameters[par]._template = self.output_template

    def _check_io_index(self, io_options: dict, update_existing = False) -> None:
        if not hasattr(self, '_index') or update_existing:
            # check that we have an output specification for the index
            if 'index' not in io_options:
                raise ValueError('No output path for index.')
            self._index = io_options['index']
            self._index._template = self.output_template

    def _set_run_options(self, run_options: dict) -> None:
        if 'history_start' in run_options and 'history_end' in run_options:
            self.reference_fn = lambda time: TimeRange(run_options['history_start'], run_options['history_end'])
        else:
            self.reference_fn = None
        
        if 'timesteps_per_year' in run_options:
            self.timesteps_per_year = run_options['timesteps_per_year']
        else:
            self.timesteps_per_year = None

#   # CLASS METHODS FOR FACTORY
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
            raise ValueError(f"Invalid data source: {index_name}")
        return Subclass
    
    @classmethod
    def get_index_name(cls, index_name: Optional[str] = None):
        if index_name is not None:
            return index_name
        elif hasattr(cls, 'index_name'):
            return cls.index_name

    #TODO: improve this, once we remove the aggregations and post-ptocessing, this can be done like cases
    def update_io(self, in_place = False, **kwargs) -> 'DRYESIndex':
        new_self = copy(self)
        new_self._raw_data = self._raw_data.update(**kwargs)
        new_self._raw_data._template = {}
        new_self.output_template = new_self._raw_data._template
        new_self._data = self._data.update(**kwargs)
        new_self._data._template = new_self.output_template
        new_self._parameters = {}
        for par in self.parameters:
            new_self._parameters[par] = self._parameters[par].update(**kwargs)
            new_self._parameters[par]._template = new_self.output_template

        new_self._index = self._index.update(**kwargs)
        new_self._index._template = new_self.output_template

        if in_place:
            self = new_self
            return self
        else:
            return new_self

    def compute(self, current:   TimeRange|Iterable[datetime],
                      reference: Optional[TimeRange|Callable[[datetime], TimeRange]] = None,
                      timesteps_per_year: Optional[int] = None) -> None:
        
        if not self._raw_data.has_tiles:
            self.compute_tile(current, reference, timesteps_per_year)
        else:
            for tile in self._raw_data.tile_names:
                new_self = self.update_io(tile = tile)
                new_self.compute_tile(current, reference, timesteps_per_year)
        
    def compute_tile(self, current:   TimeRange,
                           reference: Optional[TimeRange|Callable[[datetime], TimeRange]] = None,
                           timesteps_per_year: Optional[int] = None) -> None:
        
        if not hasattr(self, 'reference_fn') or self.reference_fn is None:
            if reference is None:
                raise ValueError('No reference period specified.')
            raw_reference = deepcopy(reference)
            # make the reference period a function of time, for extra flexibility
            if isinstance(reference, tuple) or isinstance(reference, list):
                reference_fn = lambda time: raw_reference
            elif isinstance(reference, Callable):
                reference_fn = reference
            
            self.reference_fn = reference_fn
        else:
            reference_fn = self.reference_fn
        
        if not hasattr(self, 'timesteps_per_year') or self.timesteps_per_year is None:
            if timesteps_per_year is None:
                raise ValueError('No timesteps per year specified.')
            self.timesteps_per_year = timesteps_per_year
        else:
            timesteps_per_year = self.timesteps_per_year

        if self.options['make_parameters']:
            # get the timesteps for which we need to calculate the index
            timesteps:list[FixedNTimeStep] = current.get_timesteps_from_tsnumber(timesteps_per_year)
            #create_timesteps(current.start, current.end, timesteps_per_year)
            # get the reference periods that we need to calculate parameters for
            reference_periods:list[TimeRange] = self.make_reference_periods(timesteps, reference_fn)

            # calculate the parameters
            for reference_ in reference_periods:
                self.make_parameters(reference_, timesteps_per_year)

        # calculate the index
        self.make_index(current, reference_fn, timesteps_per_year)
    
    def make_data_timesteps(self,
                            time_range: TimeRange,
                            timesteps_per_year: int) -> list[FixedNTimeStep]:
        """
        This function will return the timesteps for which the data needs to be computed.
        """
        
        if not self.iscontinuous:
            return time_range.get_timesteps_from_tsnumber(timesteps_per_year)
            #return create_timesteps(time_range.start, time_range.end, timesteps_per_year)
        else:
            # get last available timestep, if any
            last_timestep = []
            for case in self.cases['agg']:
                available_ts = self._data.get_times(time_range, **case['tags'])
                if len(available_ts) > 0:
                    last_timestep.append(self._data.get_times(time_range, **case['tags'])[-1])
            if len(last_timestep) > 0:
                # if different aggregations have different last timesteps, we need to use the earliest
                start = min(last_timestep + [time_range.start])
            else:
                start = time_range.start
            return TimeRange(start, time_range.end).get_timesteps_from_tsnumber(timesteps_per_year)
            #return create_timesteps(start, time_range.end, timesteps_per_year)
    
    def make_reference_periods(self, current_timesteps: Iterable[datetime],
                               reference_fn: Callable[[datetime], TimeRange]) -> list[TimeRange]:
        """
        This function will return the reference periods for which the parameters need to be computed.
        """

        references = set()
        for time in current_timesteps:
            this_reference = reference_fn(time)
            references.add((this_reference.start, this_reference.end))
        
        references = list(references)
        references.sort()

        references_as_tr = [TimeRange(start, end) for start, end in references]
        return references_as_tr

    def make_input_data(self, timesteps: list[FixedNTimeStep], agg_cases: Optional[list[dict]] = None) -> None:
        """
        This function will gather compute and aggregate the input data
        """

        variable_in:  Dataset = self._raw_data
        variable_out: Dataset = self._data
        
        # if there are no aggregations to compute, just get the data in the paths
        agg_cases = agg_cases or self.cases['agg']
        if len(agg_cases) == 0:
            return

        timesteps_to_compute_list = variable_out.find_times(timesteps, rev = True, cases = agg_cases)
        timesteps_to_compute_dict = {agg['name']: timestamps for agg, timestamps in zip(agg_cases, timesteps_to_compute_list)}
        self.time_aggregation.aggregate_data(timesteps_to_compute_dict, variable_in, variable_out)
    
    def make_parameters(self,
                        history: TimeRange|Iterable[datetime],
                        timesteps_per_year: int) -> None:

        # get the timesteps for which we need to calculate the parameters
        # this depends on the time aggregation step, not the index calculation step
        timesteps:list[FixedNTimeStep] = TimeRange('1900-01-01', '1900-12-31').get_timesteps_from_tsnumber(timesteps_per_year)
        # parameters need to be calculated individually for each agg case and for each opt case
        agg_cases = self.cases['agg']
        if len(agg_cases) == 0: agg_cases = [None]
        aggs_to_do = []
        for agg in agg_cases:
            for par in self._parameters.values():
                missing = par.find_times(timesteps, rev = True, **agg['tags'])
                if len(missing) > 0:
                    aggs_to_do.append(agg)
                    break
        
        if len(aggs_to_do) == 0:
            return

        if isinstance(history, tuple) or isinstance(history, list):
            history = TimeRange(history[0], history[1])

        self.log.info(f'Calculating parameters for {history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}...')

        data_timesteps:list[FixedNTimeStep] = self.make_data_timesteps(history, timesteps_per_year)
        
        # make aggregated data for the parameters
        self.make_input_data(data_timesteps, aggs_to_do)

        # get the parameters that need to be calculated
        parameters = self.parameters
        # get the output path template for the parameters
        for par in self.parameters:
            self._parameters[par].update(history_start = history.start, history_end = history.end, in_place = True)

        # get the timesteps for which we need to calculate the parameters
        # this depends on the time aggregation step, not the index calculation step
        timesteps:list[FixedNTimeStep] = TimeRange('1900-01-01', '1900-12-31').get_timesteps_from_tsnumber(timesteps_per_year)

        for agg in aggs_to_do:
            if agg is not None:
                self.log.info(f' #Aggregation {agg["name"]}:')
                agg_tags = agg['tags']
                #par_paths = {par:substitute_values(path, agg['tags']) for par, path in output_paths.items()}
                #in_path = self.input_data_path[agg["name"]]
            else:
                agg_tags = {}
                #par_paths = output_paths
                #in_path = self.input_data_path

            variable   = self._data.update(**agg_tags)
            parameters = {p: self._parameters[p].update(**agg_tags) for p in self.parameters}
            self.make_parameter_1agg(variable, parameters, history, timesteps)

    def make_parameter_1agg(self,
                            variable: Dataset,
                            parameters: dict[str:Dataset],
                            history: TimeRange,
                            timesteps: list[FixedNTimeStep]) -> dict:


            # check timesteps that have already been calculated for each parameter
            timesteps_to_do = {}
            for parname, par in parameters.items():
                this_ts_todo = par.find_times(timesteps, rev = True, cases = self.cases['opt'])
                for case in self.cases['opt']:
                    # check if this parameter is relevant for this case (this is the case for distribution parameters)
                    has_distr = 'distribution' in self.options
                    is_distr_par = has_distr and any([distr in parname for distr in self.options['distribution']])
                    is_this_distr_par = is_distr_par and case['options']['distribution'] in parname
                    if is_distr_par and not is_this_distr_par:
                        continue
                    this_ts_todo = par.find_times(timesteps, rev = True, **case['tags'])
                    for ts in this_ts_todo:
                        if ts not in timesteps_to_do:
                            timesteps_to_do[ts] = {}
                        if parname not in timesteps_to_do[ts]:
                            timesteps_to_do[ts][parname] = []
                        timesteps_to_do[ts][parname].append(case['id'])

            # if nothing needs to be calculated, skip
            if len(timesteps_to_do) == 0: return
            self.log.info(f'  -Iterating through {len(timesteps_to_do)} timesteps with missing parameters.')

            for time, par_cases in timesteps_to_do.items():
                self.log.info(f'   {time}')

                pars_data, pars_info = self.calc_parameters(time, variable, history, par_cases)

                for parname in pars_data:
                    par = parameters[parname]
                    for case, data in pars_data[parname].items():
                        tags = self.cases['opt'][case]['tags']
                        metadata = deepcopy(self.cases['opt'][case]['options'])
                        metadata.update(pars_info)
                        if 'time' in metadata: metadata.pop('time')
                        par.write_data(data, time = time, time_format = '%d/%m', metadata = metadata, **tags)

    def make_index(self, current:   TimeRange|Iterable[datetime],
                         reference: TimeRange|Iterable[datetime]|Callable,
                         timesteps_per_year: int) -> None:

        if isinstance(current, tuple) or isinstance(current, list):
            current = TimeRange(current[0], current[1])
        self.log.info(f'Calculating index for {current.start:%d/%m/%Y}-{current.end:%d/%m/%Y}...')

        timesteps:list[FixedNTimeStep] = self.make_data_timesteps(current, timesteps_per_year)
        # make aggregated data for the parameters
        self.make_input_data(timesteps)

        raw_reference = deepcopy(reference)
        # make the reference period a function of time, for extra flexibility
        if isinstance(reference, TimeRange):
            reference_fn = lambda time: raw_reference
        elif isinstance(reference, tuple) or isinstance(reference, list):
            reference_fn = lambda time: TimeRange(raw_reference[0], raw_reference[1])
        elif isinstance(reference, Callable):
            if isinstance(reference(current.start), TimeRange):
                reference_fn = lambda time: raw_reference(time.start)
            elif isinstance(reference(current.start), tuple) or isinstance(reference(current.start), list):
                reference_fn = lambda time: TimeRange(raw_reference(time.start)[0], raw_reference(time.start)[1])

        # check if anything has been calculated already
        agg_cases = self.cases['agg'] if len(self.cases['agg']) > 0 else [{}]
        for agg in agg_cases:
            agg_tags = agg['tags'] if 'tags' in agg else {}
            agg_name = agg['name'] if 'name' in agg else ''

            for case_ in self.cases['opt']:
                case = case_.copy()
                case['tags'].update(agg_tags)
                case['tags']['post_fn'] = ""

                case['name'] = case['name'] if len(agg_name) == 0 else ', '.join([f'Aggregation {agg_name}', case['name']])

                ts_todo = self._index.find_times(timesteps, rev = True, **case['tags'])

                if len(ts_todo) == 0:
                    self.log.info(f' #Case {case["name"]}: already calculated.')
                else:        
                    self.log.info(f' #Case {case["name"]}: {len(timesteps) - len(ts_todo)}/{len(timesteps)} timesteps already computed.')
                    for time in ts_todo:
                        self.log.info(f'   {time}')
                        history = reference_fn(time)
                        case['tags'].update({'history_start': history.start, 'history_end': history.end})      
                        index_data, index_info = self.calc_index(time, history, case)
                        if index_data is None:
                            index_data = self._index.build_templatearray(self._raw_data.get_template_dict())
                        metadata = case['options']
                        metadata.update(index_info)
                        if 'time' in metadata: metadata.pop('time')
                        self._index.write_data(index_data, time = time,  metadata = metadata, **case['tags'])

                # now do the post-processing
                for post_case in self.cases['post']:
                    pre_case = deepcopy(case)
                    case['tags'].update(post_case['tags'])
                    ts_todo = self._index.find_times(timesteps, rev = True, **case['tags'])
                    if len(ts_todo) == 0:
                        self.log.info(f'  Post-processing {post_case["name"]}: already calculated.')
                        continue
                    self.log.info(f'  Post-processing {post_case["name"]}: {len(timesteps) - len(ts_todo)}/{len(timesteps)} timesteps already computed.')
                    for time in ts_todo:
                        self.log.info(f'   {time}')
                        history = reference_fn(time)
                        case['tags'].update({'history_start': history.start, 'history_end': history.end})
                        index_data = self._index.get_data(time, **pre_case['tags'])
                        post_fn = post_case['post_fn']
                        ppindex_data, ppindex_info = post_fn(index_data)

                        metadata = case['options']
                        metadata.update(ppindex_info)
                        metadata.update(index_data.attrs)
                        if 'time' in metadata: metadata.pop('time')
                        self._index.write_data(ppindex_data, time = time, metadata = metadata, **case['tags'])

    def calc_parameters(self,
                        time: TimeStep,
                        variable: Dataset,
                        history: TimeRange,
                        par_and_cases: dict[str:list[int]]) -> tuple[dict[str:dict[int:np.ndarray]], dict]:
        """
        Calculates the parameters for the index.
        par_and_cases is a dictionary with the following structure:
        {par: [case1, case2, ...]}
        indicaing which cases from self.cases['opt'] need to be calculated for each parameter.

        2 outputs are returned:
        - a dictionary with the following structure:
            {par: {case1: parcase1, case2: parcase1, ...}}
            where parcase1 is the parameter par for case1 as a numpy.ndarray.
        - a dictionary with info about parameter calculations.
        """
        raise NotImplementedError
    
    def calc_index(self, time: TimeStep,  history: TimeRange, case: dict) -> tuple[np.ndarray, dict]:
        """
        Calculates the index for the given time and reference period.
        Returns the index as a numpy.ndarray and a dictionary of metadata, if any.
        """
        raise NotImplementedError