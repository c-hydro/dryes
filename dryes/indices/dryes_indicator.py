from datetime import datetime
from typing import Iterable
from copy import deepcopy

from dryes.tools.timestepping.fixed_num_timestep import FixedNTimeStep
from dryes.tools.timestepping.timerange import TimeRange

from .dryes_index import DRYESIndex
from ..index_combination import COMBINATION_ALGORITHMS

class DRYESCombinedIndicator(DRYESIndex):
    index_name = 'indicator'
    default_options = {
        'algorithm': 'cdi_jrc',
    }

    @property
    def parameters(self):
        input_data, previous_data, output_data = self.algorithm()
        input_keys = list(k for k in input_data)
        previous_keys = list(k for k in previous_data)
        output_keys = list(k for k in output_data)
        return input_keys + previous_keys + output_keys[1:]

    def _check_index_options(self, options: dict) -> None:
        super()._check_index_options(options)
        self.algorithm = COMBINATION_ALGORITHMS.get(self.options.pop('algorithm'))

    def _check_io_options(self, io_options: dict, update_existing = False) -> None:

        # get a template from any of the input datasets
        io_names = self.parameters
        for name in io_names:
            ds       = io_options.get(name)
            template = ds.get_template(**self.cases['opt'][0]['options'])
            if template is not None:
                self.output_template = template
                break

        # check that we have all the necessary options
        self._check_io_parameters(io_options, update_existing)
        self._check_io_index(io_options, update_existing)

    def make_input_data(self, timesteps: list[FixedNTimeStep]):
        # input data should be done already (i.e. no need to aggregate anything)
        pass

    def make_parameters(self,
                        history: TimeRange|Iterable[datetime],
                        timesteps_per_year: int) -> None:
        
        self.log.warning('No parameters to calculate for this index.')
        return

    def calc_index(self,
                   time: FixedNTimeStep,
                   case: dict) -> tuple:
        
        input_data, previous_data, output_data = self.algorithm()
        input_keys = list(k for k in input_data)
        previous_keys = list(k for k in previous_data)

        this_input_data    = {k:self._parameters[k].get_data(time, **case['tags']) for k in input_keys}
        this_previous_data = {k:self._parameters[k].get_data(time-1, **case['tags']) for k in previous_keys}

        this_output_data = self.algorithm(this_input_data, this_previous_data)
        self.save_output_data(this_output_data, time, case['options'], **case['options'])

        index_name = output_data[0]
        index_data = this_output_data[index_name]
        index_info = case['options']

        return index_data, index_info
    
    def save_output_data(self,
                         data: dict,
                         time: FixedNTimeStep,
                         metadata: dict, **tags) -> None:
        
        _, _, ouptut_data = self.algorithm()
        output_keys = list(k for k in ouptut_data)

        for key in output_keys:
            if key not in self._parameters:
                continue
            this_ds = self._parameters[key]
            this_ds.write_data(data[key], time = time, metadata = metadata, **tags)

    def make_index(self,
                   current:   TimeRange|Iterable[datetime],
                   timesteps_per_year: int) -> None:

        if isinstance(current, tuple) or isinstance(current, list):
            current = TimeRange(current[0], current[1])
        self.log.info(f'Calculating index for {current.start:%d/%m/%Y}-{current.end:%d/%m/%Y}...')

        timesteps:list[FixedNTimeStep] = self.make_data_timesteps(current, timesteps_per_year)

        # check if anything has been calculated already
        for case_ in self.cases['opt']:
            case = case_.copy()
            case['tags']['post_fn'] = ""
            case['name'] = case['name']

            ts_todo = self._index.find_times(timesteps, rev = True, **case['tags'])

            if len(ts_todo) == 0:
                self.log.info(f' #Case {case["name"]}: already calculated.')
            else:        
                self.log.info(f' #Case {case["name"]}: {len(timesteps) - len(ts_todo)}/{len(timesteps)} timesteps already computed.')
                for time in ts_todo:
                    self.log.info(f'   {time}')     
                    index_data, index_info = self.calc_index(time, case)
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
                        index_data = self._index.get_data(time, **pre_case['tags'])
                        post_fn = post_case['post_fn']
                        ppindex_data, ppindex_info = post_fn(index_data)

                        metadata = case['options']
                        metadata.update(ppindex_info)
                        metadata.update(index_data.attrs)
                        if 'time' in metadata: metadata.pop('time')
                        self._index.write_data(ppindex_data, time = time, metadata = metadata, **case['tags'])