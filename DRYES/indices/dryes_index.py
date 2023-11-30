from datetime import datetime
from typing import Callable, List, Optional
import xarray as xr
import os
import numpy as np

from ..variables.dryes_variable import DRYESVariable
from ..time_aggregation.time_aggregation import TimeAggregation

from ..lib.log import setup_logging, log
from ..lib.time import TimeRange, create_timesteps, ntimesteps_to_md
from ..lib.parse import substitute_values, options_to_cases
from ..lib.io import check_data_range, save_dataarray_to_geotiff, get_data

class DRYESIndex:
    def __init__(self, input_variable: DRYESVariable,
                 timesteps_per_year: int,
                 options: dict,
                 output_paths: dict,
                 log_file: str = 'DRYES_log.txt') -> None:
        
        setup_logging(log_file)

        self.input_variable = input_variable
        self.timesteps_per_year = timesteps_per_year

        self.check_options(options)
        self.get_cases()

        self.output_paths    = substitute_values(output_paths, output_paths, rec = False)
        self.output_template = input_variable.grid.template
    
    def check_options(self, options: dict) -> dict:
        these_options = self.default_options.copy()
        these_options.update({'post_fn': None})
        for k in these_options.keys():
            if k in options:
                these_options[k] = options[k]
            else:
                log(f'No option {k} specified, using default value: {these_options[k]}.')

        # all of this is a way to divide the options into three bits:
        # - the time aggregation, which will affect the input data (self.time_aggregation)
        # - the options that will be used to calculate the parameters and the index (self.options)
        # - the post-processing function, which will be applied to the index (self.post_processing)
        options = self.make_time_aggregation(these_options)
        self.post_processing = options['post_fn']
        del options['post_fn']
        self.options = options

    def make_time_aggregation(self, options: dict) -> dict:
        # if the iscontinuous flag is not set, set it to False
        if not hasattr(self, 'iscontinuous'):
                self.iscontinuous = False

        # deal with the time aggregation options
        if not 'agg_fn' in options:
            self.time_aggregation = TimeAggregation(365)
            return options
        
        time_agg = TimeAggregation(self.timesteps_per_year)
        for agg_name, agg_fn in options['agg_fn'].items():
            if isinstance(agg_fn, Callable):
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

    def get_cases(self) -> None:

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
            this_case['tags'] = {'options.agg_fn': agg_name}
            agg_cases.append(this_case)

        ## then add the options
        opt_cases = options_to_cases(options)

        ## finally add the post-processing, if it exists
        post_cases = []
        if post_processing is not None:
            for post_name, post_fn in post_processing.items():
                this_case = dict()
                this_case['id']   = i
                this_case['name'] = post_name
                this_case['tags'] = {'options.post_fn': post_name}
                this_case['post_fn'] = post_fn
                post_cases.append(this_case)

        ## and combine them
        self.cases = {'agg': agg_cases, 'opt': opt_cases, 'post': post_cases}

    def compute(self, current: TimeRange, reference: TimeRange|Callable[[datetime], TimeRange]) -> None:
        
        # make the reference period a function of time, for extra flexibility
        if isinstance(reference, TimeRange):
            reference_fn = lambda time: reference
        elif isinstance(reference, Callable):
            reference_fn = reference

        # get the timesteps for which we need input data
        data_timesteps = self.make_data_timesteps(current, reference_fn)

        # get the data,
        # these will gather and compute the input data (checking if it is already available)
        # if we need a time aggregation, this will be done in the input variable
        input_data_path = self.make_input_data(data_timesteps)
        self.input_data_path = input_data_path

        # get the reference periods that we need to calculate parameters for
        reference_periods = self.make_reference_periods(current, reference_fn)

        # calculate the parameters
        for reference in reference_periods:
            self.make_parameters(reference)

        # calculate the index
        self.make_index(current, reference_fn)
    
    def make_data_timesteps(self, current: TimeRange, reference_fn: Callable[[datetime], TimeRange]) -> List[datetime]:
        """
        This function will return the timesteps for which the data needs to be computed.
        """

        agg_timesteps_per_year = self.time_aggregation.timesteps_per_year

        # all of this will get the range for the data that is needed
        current_timesteps = create_timesteps(current.start, current.end, agg_timesteps_per_year)
        reference_start = set(reference_fn(time).start for time in current_timesteps)
        reference_end   = set(reference_fn(time).end   for time in current_timesteps)
        
        if self.iscontinuous:
            time_start = min(reference_start)
            time_end   = max(current_timesteps)
            return create_timesteps(time_start, time_end, agg_timesteps_per_year)
        else:
            reference_timesteps = create_timesteps(min(reference_start), max(reference_end), self.timesteps_per_year)
            return reference_timesteps + current_timesteps

    def make_reference_periods(self, current: TimeRange, reference_fn: Callable[[datetime], TimeRange]) -> List[TimeRange]:
        """
        This function will return the reference periods for which the parameters need to be computed.
        """

        # all of this will get the range for the data that is needed
        current_timesteps = create_timesteps(current.start, current.end, self.timesteps_per_year)
        references = set()
        for time in current_timesteps:
            this_reference = reference_fn(time)
            references.add((this_reference.start, this_reference.end))
        
        references = list(references)
        references.sort()

        references_as_tr = [TimeRange(start, end) for start, end in references]
        return references_as_tr
    
    def make_input_data(self, timesteps: List[datetime]) -> dict[str:str]:
        """
        This function will gather compute and aggregate the input data
        """
        variable = self.input_variable
        time_agg = self.time_aggregation
        
        # the time aggregator recognises only the 'agg_name' keyword
        # for the path name
        path_raw = substitute_values(self.output_paths['data'], {'var': variable.name})
        self.output_paths['data'] = path_raw
        path = substitute_values(self.output_paths['data'], {'options.agg_fn': '{agg_name}'})
        
        log(f'Making input data ({variable.name})...')
        # if there are no aggregations to compute, just get the data in the paths
        agg_cases = self.cases['agg']
        if len(agg_cases) == 0:
            variable.path = path
            variable.make(TimeRange(min(timesteps), max(timesteps)))
            return path

        # get the names of the aggregations to compute
        agg_names = [case['name'] for case in agg_cases]
        agg_paths = {agg_name: path.format(agg_name = agg_name) for agg_name in agg_names}

        time_range = TimeRange(min(timesteps), max(timesteps))
        available_timesteps = {name:list(check_data_range(path, time_range)) for name,path in agg_paths.items()}

        timesteps_to_compute = {}
        for agg_name in agg_names:
            # if there is no post aggregation function, we don't care for the order of the timesteps
            if agg_name not in time_agg.postaggfun.keys():
                timesteps_to_compute[agg_name] = [time for time in timesteps if time not in available_timesteps[agg_name]]
            # if there is a post aggregation function, we need to compute the timesteps in order
            else:
                timesteps_to_compute[agg_name] = []
                i = 0
                while(timesteps[i] not in available_timesteps[agg_name]) and i < len(timesteps):
                    timesteps_to_compute[agg_name].append(timesteps[i])
                    i += 1

        timesteps_to_iterate = set.union(*[set(timesteps_to_compute[agg_name]) for agg_name in agg_names])
        timesteps_to_iterate = list(timesteps_to_iterate)
        timesteps_to_iterate.sort()

        agg_data = {n:[] for n in agg_names}
        for i, time in enumerate(timesteps_to_iterate):
            log(f'Computing {time:%d-%m-%Y} ({i+1}/{len(timesteps_to_iterate)})...')
            for agg_name in agg_names:
                log(f'#Starting aggregation {agg_name}...')
                if time in timesteps_to_compute[agg_name]:
                    agg_data[agg_name].append(time_agg.aggfun[agg_name](variable, time))
        
        data_template = self.output_template
        for agg_name in agg_names:
            log(f'#Completing time aggregation: {agg_name}...')
            if agg_name in time_agg.postaggfun.keys():
                agg_data[agg_name] = time_agg.postaggfun[agg_name](agg_data[agg_name], variable)

            n = 0
            for data in agg_data[agg_name]:
                this_time = timesteps_to_compute[agg_name][i]
                path_out = this_time.strftime(agg_paths[agg_name])
                output = data_template.copy(data = data.values)
                metadata = {'name' : variable.name,
                            'time' : this_time,
                            'type' : 'DRYES data',
                            'index': self.index_name,
                            'aggregation': agg_name}
                
                save_dataarray_to_geotiff(data, path_out, metadata)
                n += 1
            
            log(f'#Saved {n} files to {os.path.dirname(self.output_paths['data'])}.')

        return agg_paths
    
    def make_parameters(self, history: TimeRange):
        log(f'Calculating parameters for {history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}...')
        
        # get the output path template for the parameters
        output_path = self.output_paths['parameters']
        output_path = substitute_values(output_path, {'history_start': history.start, "history_end": history.end})
        
        # get the parameters that need to be calculated
        parameters = self.parameters

        # get the timesteps for which we need to calculate the parameters
        # this depends on the time aggregation step, not the index calculation step
        md_timesteps = ntimesteps_to_md(self.time_aggregation.timesteps_per_year)
        timesteps = [datetime(1900, month, day) for month, day in md_timesteps]
        time_range = TimeRange(min(timesteps), max(timesteps))
        # the year for time_range and timesteps is fictitious here, parameters don't have a year.

        history_years = list(range(history.start.year, history.end.year + 1))

        # parameters need to be calculated individually for each agg case
        agg_cases = self.cases['agg']
        if len(agg_cases) == 0: agg_cases = [None]
        for agg in agg_cases:
            if agg is not None:
                log(f' #{agg["name"]}:')
                this_output_path = substitute_values(output_path, agg['tags'])
            else:
                this_output_path = output_path

            par_paths = {par: substitute_values(this_output_path, {'par': par}) for par in parameters}
            timesteps_to_do_set = {par: set() for par in parameters}
            timesteps_to_do = {par: [] for par in parameters}
            # check if anything has been calculated already
            for case in self.cases['opt']:
                case_par_paths = {par: substitute_values(par_paths[par], case['tags']) for par in parameters}
                case_done_timesteps = {par: list(check_data_range(par_path, time_range)) for par, par_path in case_par_paths.items()}
                case_timesteps_to_do = {par: [time for time in timesteps if time not in case_done_timesteps[par]] for par in parameters}
                for par in parameters:
                    timesteps_to_do_set[par].update(case_timesteps_to_do[par])
                    timesteps_to_do[par].append(case_timesteps_to_do[par])
            
            ndone = {par: len(timesteps) - len(timesteps_to_do_set[par]) for par in parameters}
            for par in parameters:
                log(f'  - {par}: {ndone[par]}/{len(timesteps)} timesteps already computed.')

            timesteps_to_iterate = set.union(*[timesteps_to_do_set[par] for par in parameters])
            timesteps_to_iterate = list(timesteps_to_iterate)
            timesteps_to_iterate.sort()

            # if nothing needs to be calculated, skip
            if len(timesteps_to_iterate) == 0: continue
            log(f' #Iterating through {len(timesteps_to_iterate)} timesteps with missing parameters.')
            for time in timesteps_to_iterate:
                month = time.month
                day   = time.day
                log(f'  - {day:02d}/{month:02d}')
                this_date = datetime(1900, month, day)
                parcases_to_calc = {par:[] for par in parameters}
                for par in parameters:
                    parcases_to_calc[par] = [i for i,dates in enumerate(timesteps_to_do[par]) if this_date in dates]

                all_dates  = [datetime(year, month, day) for year in history_years]
                data_dates = all_dates
                data_dates = [date for date in all_dates if date >= history.start and date <= history.end]

                data_template = self.output_template
                par_data = self.calc_parameters(data_dates, parcases_to_calc, history)
                for par in par_data:
                    for id, data in par_data[par].items():
                        path = substitute_values(par_paths[par], self.cases['opt'][id]['tags'])
                        path = time.strftime(path)
                        output = data_template.copy(data = data.values)
                        metadata = {'reference_period': f'{history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}',
                                    'name': par,
                                    'time': time.strftime('%d/%m'),
                                    'type': 'DRYES parameter',
                                    'index': self.index_name}
                        saved = save_dataarray_to_geotiff(output, path, metadata)

    def make_index(self, current: TimeRange, reference_fn: Callable[[datetime], TimeRange]) -> str:
        log(f'Calculating index for {current.start:%d/%m/%Y}-{current.end:%d/%m/%Y}...')

        # get the timesteps for which we need to calculate the index
        timesteps = create_timesteps(current.start, current.end, self.timesteps_per_year)

        # check if anything has been calculated already
        for agg in self.cases['agg']:
            agg_tags = agg['tags'] if 'tags' in agg else {}
            agg_name = agg['name'] if 'name' in agg else ''

            for case_ in self.cases['opt']:
                case = case_.copy()
                case['tags'].update(agg_tags)
                case['tags']['options.post_fn'] = ""

                case['name'] = case['name'] if len(agg_name) == 0 else ', '.join([agg_name, case['name']])

                this_index_path_raw = substitute_values(self.output_paths['maps'], {'index': self.index_name})
                this_index_path = substitute_values(this_index_path_raw, case['tags'])
                done_timesteps = check_data_range(this_index_path, current)
                timesteps_to_compute = [time for time in timesteps if time not in done_timesteps]
                if len(timesteps_to_compute) == 0:
                    log(f' - case {case["name"]}: already calculated.')
                else:        
                    log(f' - case {case["name"]}: {len(timesteps) - len(timesteps_to_compute)}/{len(timesteps)} timesteps already computed.')
                    for time in timesteps_to_compute:
                        log(f'  - {time:%d/%m/%Y}')
                        reference = reference_fn(time)
                        index = self.calc_index(time, reference, case)
                        output = self.output_template.copy(data = index.values)
                        metadata = {'name': self.index_name,
                                    'time': time,
                                    'type': 'DRYES index',
                                    'index': self.index_name}
                        path_out = time.strftime(this_index_path)
                        save_dataarray_to_geotiff(output, path_out, metadata)

                # now do the post-processing
                for post_case in self.cases['post']:
                    case['tags'].update(post_case['tags'])
                    this_ppindex_path = substitute_values(this_index_path_raw, case['tags'])
                    done_timesteps = check_data_range(this_ppindex_path, current)
                    timesteps_to_compute = [time for time in timesteps if time not in done_timesteps]
                    if len(timesteps_to_compute) == 0:
                        log(f' - post-processing {post_case["name"]}: already calculated.')
                        continue

                    log(f'  - post-processing {post_case["name"]}: {len(timesteps) - len(timesteps_to_compute)}/{len(timesteps)} timesteps already computed.')
                    for time in timesteps_to_compute:
                        log(f'   - {time:%d/%m/%Y}')
                        reference = reference_fn(time)
                        index = get_data(this_index_path, time)
                        post_fn = post_case['post_fn']
                        ppindex_data = post_fn(index.values)
                        output = self.output_template.copy(data = np.expand_dims(ppindex_data, axis = 0))
                        metadata = {'name': self.index_name,
                                    'time': time,
                                    'type': 'DRYES index',
                                    'index': self.index_name,
                                    'post-processing': post_case['name']}
                        path_out = time.strftime(this_ppindex_path)
                        save_dataarray_to_geotiff(output, path_out, metadata)

    def calc_parameters(self, dates: List[datetime],
                        par_and_cases: dict[str:List[int]],
                        reference: Optional[TimeRange]=None) -> dict[str:dict[int:xr.DataArray]]:
        """
        Calculates the parameters for the index.
        par_and_cases is a dictionary with the following structure:
        {par: [case1, case2, ...]}
        indicaing which cases from self.cases['opt'] need to be calculated for each parameter.

        The output is a dictionary with the following structure:
        {par: {case1: parcase1, case2: parcase1, ...}}
        where parcase1 is the parameter par for case1 as a xarray.DataArray.
        """
        raise NotImplementedError
    
    def calc_index(self, time,  reference: TimeRange, case: dict) -> xr.DataArray:
        """
        Calculates the index for the given time and reference period.
        """
        raise NotImplementedError