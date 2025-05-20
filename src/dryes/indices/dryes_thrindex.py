from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import tempfile
import shutil
import copy
import warnings

from .dryes_index import DRYESIndex

import d3tools.timestepping as ts
from d3tools.cases import CaseManager

from ..core.threshold_indices import calc_thresholds_cdo, pool_index, calc_dintensity

class DRYESThrBasedIndex(DRYESIndex):
    """
    This class implements threshold based indices.
    The indices that are relevant for this class are: LFI (Low Flow Index) and HCWI (Heat and Cold Wave Index).
    """

    index_name = 'Threshold based index'

    # The way this index works is a little different from the other indices.
    # Here, we do two separate steps:
    # 1. Calculate the "daily" index, which is the difference between the data and the threshold.
    # 2. Pool the daily index to get the final index.

    # The daily index is calculated in the same way as the other indices, the pooling is unique to this class.
    # The daily index determines, for each day
    #  (1) if it is a hit day or not
    #  (2) the intensity of the day -> unless different in the subclass a day is a hit day if the intensity is > 0

    # The pooling is unique to this class and determines if a sequence of hit/non-hit days constitutes a spell, an event or not.
    # The output of the pooling is a set of variables that describe the current condition:
    #  - intensity : the intensity of the event, which is the sum of the intensities of the days in the event
    #  - duration  : the duration of the event, which is the number of days in the event
    #  - interval  : the interval since the last spell, which is the number of days since the last hit day (0 if a hit day)
    #  - nhits     : the number of hit days in the spell (0 if not a spell)
    #  - case      : a debugging variable (see below)
    # intensity and duration are the actual variables of interest -> refer only to events, not to spells!
    # interval and nhits are used to determine the end/start of a spell and if a spell is an event or not
    # case is a debugging variable that is used to determine the type of day:
    #  - 00 : ERROR! THIS SHOULD NOT HAPPEN!
    #  - 01 : not a hit day that is not part (or the end of) a spell
    #  - 02 : pool day that is part of an event: min_duration already reached (only if count_with_pools is True)
    #  - 03 : pool day that will be part of an event: min_duration will be reached in the future (only if count_with_pools is True and look_ahead is True)
    #  - 04 : pool day that is not part of an event: min_duration is not reached and will not be reached in the future (if count_with_pools is False, this is any pool day)
    #  - 05 : not hot day that could be a pool day (we don't know yet)
    #  - 06 : not hot day that is the end of a spell (reset the counters)
    #  - 12 : hot day that is part of an event: min_duration already reached
    #  - 13 : hot day that will be part of an event: min_duration will be reached in the future (only if look_ahead is True)
    #  - 14 : hot day that is not part of an event: min_duration is not reached and will not be reached in the future
    #  - 15 : hot day that is not yet part of an event: min_duration is not reached, but we don't know yet if it will be reached in the future

    ## In the comments, and the naming of the variables, we will use the following convention:
    ## - 'hit' day  : a day that is above the threshold
    ## - 'spell'    : a sequence of hit days separated by at most "min_interval" pool days
    ## - 'pool' day : a day that is not a hit day, but is part of a spell, there can by a maximum of "min_interval" pool days between two hit days
    ## - 'event'    : a spell that is longer than the "minimum_duration"

    ## Special options:
    # - look_ahead: True/False
    #     if True, the algorithm will look a a number of days in the future to determine:
    #     - if a non-hit day after a spell is a pool day or not (interval + future_interval <= min_interval)
    #     - if a spell that is not an event will become an event in the future (reaching nhits >= min_duration)
    #     The number of days to look ahead is
    #       - (min_duration-1)*(min_interval+1) if min_duration >  1, this is the maximum number of days that a spell can be without being an event
    #       - min_interval                      if min_duration == 1
    #     If not enough data are available to look ahead, the algorithm will quietly look aheas as much as it can and mark the final result as "PRELIMINARY"
    # - pool_if_uncertain: True/False
    #     if True, the algorithm will consider a non-hit day after a spell as a pool day unless it is certain that it is not
    #     if look_ahead is False, this is the same as saying that all spells continue for min_interval days after the last hit day
    #     if look_ahead is True, this is only relevant if there is not enough data to look min_interval days ahead
    # - count_with_pools: True/False
    #     if True, the intensity of pool days will be included in the final intensity if the pool day is part of an event, and pool days will count towards the event duration.

    # default options
    default_options = {
        'thr_quantile' :  0.1,   # quantile for the threshold calculation
        'thr_window'   :  1,     # window size for the threshold calculation (n of days)
        'min_interval' :  1,     # minimum interval between spells
        'min_duration' :  5,     # minimum number of hit days in a spell to be considered an event (exluding pool days!)
        
        'look_ahead'        : True,   # if True, tries to look ahead to see if a spell continues, look-ahead time is (min_duration-1)*(min_interval+1)
        'count_with_pools'  : True,   # if True, the pool days are added to the duration and intensity of the spell
        'pool_if_uncertain' : 1,      # 0: False, reset the counters;
                                      # 1: True, but don't increase intensity and duration even if count_with_pools;
                                      # 2: True, increase intensity and duration if count_with_pools.

        'cdo_path'     :  '/usr/bin/cdo', # path to the cdo executable, to calculate the thresholds

        # options to set the tags of the output data
        'pooled_var_name': 'var',
        'pooled_vars'    : {'intensity'  : 'intensity',      # intensity of the event
                            'duration'   : 'duration',       # duration of the event
                            'interval'   : 'interval',       # interval since the last hit day
                            'nhits'      : 'nhits',          # number of hit days in the spell
                            'sintensity' : 'sintensity',     # intensity of the spell (incl. uncertain pool days if count_with_pools is True)
                            'sduration'  : 'sduration',      # duration of the spell  (incl. uncertain pool days if count_with_pools is True)
                            'case'       : 'case'}           # debugging case of the output
    }

    option_cases = {
        'parameters_thr' : ['thr_quantile', 'thr_window', 'cdo_path'],
        'index_daily'    : []
    }

    option_cases_pooling = {
        'index_pooled' : ['min_interval', 'min_duration', 'look_ahead', 'pool_if_uncertain', 'count_with_pools'],
    }

    parameters = ['threshold']
    parameters_pooling = []

    @property
    def direction(self):
        # this is the direction of the index, -1 for deficits (LFI, Cold waves), 1 for surplusses (Heat waves)
        # must be implemented in the subclass
        raise NotImplementedError

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # get max look-ahead time
        look_ahead = self.options.get('look_ahead')
        if isinstance(look_ahead, dict):
            look_ahead = any(look_ahead.values())
        
        if look_ahead:
            min_interval = self.options.get('min_interval')
            if isinstance(min_interval, dict):
                min_interval = max(min_interval.values())
            min_duration = self.options.get('min_duration')
            if isinstance(min_duration, dict):
                min_duration = max(min_duration.values())
            self.max_look_ahead = (min_duration - 1) * (min_interval + 1) if min_duration > 1 else min_interval
        else:
            self.max_look_ahead = 0

    def _set_io_tags(self) -> None:

        self.pooled_var_name = self.options.pop('pooled_var_name')
        self.pooled_vars     = self.options.pop('pooled_vars')

    def _check_io_data(self, io_options, update_existing=False):
        self._set_io_tags()
        super()._check_io_data(io_options, update_existing)

    def _check_io_parameters(self, io_options: dict, update_existing = False) -> None:
        super()._check_io_parameters(io_options, update_existing)
        for par in self.parameters_pooling:
            if par not in io_options: raise ValueError(f'No source/destination specified for parameter {par}.')
            self._parameters[par] = io_options[par]
            self._parameters[par]._template = self.output_template

    def _check_io_index(self, io_options, update_existing=False):

        if not hasattr(self, '_index_pooled') or update_existing:
            # check that we have an output specification for the index
            if 'index' not in io_options: raise ValueError('No output path specified for the index.')
            self._index_pooled = io_options['index']
            self._index_pooled._template = self.output_template

        if not hasattr(self, '_index') or update_existing:
            # check that we have an output specification for the daily components of the index
            self._index = io_options.get('index_daily')
            if self._index is None:
                from d3tools.data.memory_dataset import MemoryDataset
                key_pattern = self._index_pooled.key_pattern.replace('.tif', '_daily.tif')
                self._index = MemoryDataset(key_pattern)
                if self._index.has_tiles: self._index.tile_names = self._data.tile_names
            
            self._index._template = self.output_template

    def _set_cases(self) -> None:
        self.cases, self.cases_pooling = self._get_cases()

    def _get_cases(self) -> dict:
        cases = super()._get_cases()
        options = self.options
        
        last_options  = cases.options[-1]
        cases_pooling = CaseManager(last_options, 'index_daily')

        for layer, opts in self.option_cases_pooling.items():
            these_options = {k:options.get(k, self.default_options.get(k)) for k in opts}
            cases_pooling.add_layer(these_options, layer)

        return cases, cases_pooling

    def _make_parameters(self, history: ts.TimeRange, frequency: str|None) -> None:

        for data_case_id, data_case in self.cases['data'].items():       
            self._make_thresholds(history, data_case, data_case_id)

        # get any other parameters
        other_par = [p for p in self.parameters if p != 'threshold']
        if len(other_par) > 0:
            self._make_other_parameters(history)

    def _make_thresholds(self, history: ts.TimeRange, data_case: dict, data_case_id: str, var = None, var_tags = None) -> None:
        
        var      = var      or self._data
        var_tags = var_tags or {}

        days = history.days
        
        data_case.options.update(var_tags)
        tmpdir = tempfile.mkdtemp()

        # Create a temporary NetCDF file to store the concatenated data
        data_nc = f'{tmpdir}/data.nc'

        import netCDF4

        da0 = var.get_data(days[0], **data_case.options).squeeze().expand_dims("time").assign_coords(time=[days[0].start])
        ds0: xr.Dataset = da0.to_dataset(name="data")
        ds0.to_netcdf(data_nc, mode="w", unlimited_dims = ['time'], encoding={'data': {'zlib' : True, 'complevel': 4}})
        unit = f'days since {days[0].start:%Y-%m-%d}'

        # Append data incrementally
        for day in days[1:]:
            da = var.get_data(day, as_is = True, **data_case.options).squeeze().expand_dims("time").assign_coords(time=[day.start])
            with netCDF4.Dataset(data_nc, mode="a") as ncfile:
                time_index = len(ncfile.variables["time"])
                ncfile.variables["time"][time_index] = netCDF4.date2num(day.start, units=unit)
                ncfile.variables["data"][time_index, :, :] = da.values

        # get the timesteps for which we need to calculate the parameters - thresholds are always daily (incl. 29th Feb)
        timesteps:list[ts.TimeStep] = ts.TimeRange('1904-01-01', '1904-12-31').days

        # loop through the cases for the thresholds
        thr_cases = self.cases['parameters_thr']
        for thr_case_id, thr_case in thr_cases.items():
            if not thr_case_id.startswith(data_case_id):
                continue

            thr_quantile = thr_case.options['thr_quantile']
            thr_window   = thr_case.options['thr_window']
            cdo_path     = thr_case.options['cdo_path']
            thresholds = calc_thresholds_cdo(data_nc, thr_quantile, thr_window, cdo_path, var_name = 'data')

            for time, thr_data in zip(timesteps, thresholds):
                metadata = thr_case.options.copy()
                metadata.update({'reference': f'{history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}'})
                tags = thr_case.tags | var_tags
                self._parameters['threshold'].write_data(thr_data, time = time, time_format = '%d/%m', metadata = metadata, **tags)

        shutil.rmtree(tmpdir)

    def _make_index(self, current: ts.TimeRange, reference: ts.TimeRange, frequency: str) -> None:
       
        # adjust the current considering we might need some look-ahead time
        extended_current = current.extend(ts.TimeWindow(self.max_look_ahead, 'd'))

        ## figure out the last daily index, to adjust current if needed
        last_daily  = self.get_last_ts_index_daily(now = extended_current.end, lim = current.start)

        # make the daily index here:
        if last_daily is None: last_daily = ts.Day.from_date(current.start) - 1
        if last_daily.end < extended_current.end:
            daily_tr = ts.TimeRange(last_daily.start + timedelta(days = 1), extended_current.end)
            super()._make_index(daily_tr, reference, 'd')

        # and then pool it!
        self._make_index_pooled(current, reference, frequency)

    def _make_index_pooled(self, current: ts.TimeRange, reference: ts.TimeRange, frequency: str) -> None:
        
        for index_daily_case_id, index_daily_case in self.cases_pooling['index_daily'].items():

            if frequency is None:
                data_ts_unit = self._data.estimate_timestep(**index_daily_case.options).unit
                frequency = data_ts_unit

            # get the timesteps for which we need to calculate the index
            timesteps:list[ts.TimeStep] = current.get_timesteps(frequency)

            # loop through all the timesteps
            for time in timesteps:

                # get the daily index for this time and (if needed) the look-ahead period
                if self.max_look_ahead == 0:
                    daily_index, metadata = self.get_index_daily(ts.TimeRange(time.start, time.end), index_daily_case)
                    iscomplete = True
                else:
                    end_time = time.end + timedelta(days = self.max_look_ahead)
                    daily_index, metadata = self.get_index_daily(ts.TimeRange(time.start, end_time), index_daily_case)
                    iscomplete = daily_index['dintensity'] is not None and daily_index['dintensity'].shape[0] >= self.max_look_ahead

                for idx_case_id, idx_case in self.cases_pooling[-1].items():
                    if not idx_case_id.startswith(index_daily_case_id):
                        continue
                    
                    previous_index, _ = self.get_index_pooled(time - 1, idx_case)
                    this_daily_index = copy.deepcopy(daily_index)

                    this_index = self.pool_index(this_daily_index, previous_index,
                                                 options = idx_case.options,
                                                 n = time.length())

                    metadata.update(idx_case.options.copy())
                    metadata.update({'reference': f'{reference.start:%d/%m/%Y}-{reference.end:%d/%m/%Y}'})
                    metadata.update({'COMPLETE': str(iscomplete).lower()})

                    tags = idx_case.tags

                    for index, other_tags in this_index:
                        self._index_pooled.write_data(index, time = time, metadata = metadata, **tags, **other_tags)

    def calc_index(self,
                   data: np.ndarray, parameters: dict[str, np.ndarray],
                   options: dict, step = 1, **kwargs) -> np.ndarray:

        thresholds = parameters['threshold']
        return calc_dintensity(data, thresholds, self.direction, get_hits=False)['dintensity']
    
    def pool_index(self,
                   daily_index: dict[str:np.ndarray], 
                   current_pooled_index: dict[str:np.ndarray],
                   options: dict, n: int = 1) -> tuple[np.ndarray,dict]:

        pooled_index = pool_index(daily_index, current_pooled_index, n = n,
                                  min_interval = options['min_interval'],
                                  min_duration = options['min_duration'],
                                  look_ahead = options['look_ahead'],
                                  pool_if_uncertain = options['pool_if_uncertain'],
                                  count_with_pools = options['count_with_pools'])

        output = []

        for key, value in pooled_index.items():
            output.append((value, {self.pooled_var_name: self.pooled_vars[key]}))

        return output

    def get_index_daily(self, time: ts.TimeRange, case) -> tuple[dict[str:np.ndarray], dict]:
        
        if not isinstance(time, ts.Day):
            days_in_time = time.days
            all_dintensity_np = []
            all_ishit_np      = []
            all_metadata      = {}
            for t in days_in_time:
                this_daily_index, this_metadata = self.get_index_daily(t, case)
                this_dintensity_np = this_daily_index['dintensity']
                this_ishit_np      = this_daily_index['ishit']
                if this_dintensity_np is None or this_ishit_np is None:
                    continue ##TODO: ADD A WARNING OR SOMETHING
                all_dintensity_np.append(this_dintensity_np)
                all_ishit_np.append(this_ishit_np)

                for key in self.options['propagate_metadata']:
                    if key in this_metadata:
                        if key not in all_metadata:
                            all_metadata[key] = [this_metadata[key]]
                        else:
                            all_metadata[key].append(this_metadata[key])

            all_metadata = {k:','.join(v) for k,v in all_metadata.items()}
            all_metadata.update(case.options.copy())

            if len(all_dintensity_np) == 0:
                return {'dintensity' : None, 'ishit' : None}, {}
            
            all_dintensity_np = np.stack(all_dintensity_np)
            all_ishit_np      = np.stack(all_ishit_np)

            return {'dintensity' : all_dintensity_np, 'ishit' : all_ishit_np}, all_metadata

        if not self._index.check_data(time, **case.tags):
            return {'dintensity' : None, 'ishit' : None}, {}

        dintensity = self._index.get_data(time, **case.tags)
        ishit      = xr.where(dintensity > 0, 1, 0).astype('int8')

        metadata = {}
        for key in self.options['propagate_metadata']:
            if key in dintensity.attrs:
                metadata[key] = dintensity.attrs[key]
        metadata.update(case.options.copy())

        return {'dintensity' : dintensity.values.squeeze(), 'ishit' : ishit.values.squeeze()}, metadata

    def get_parameters(self, time: ts.TimeStep, case) -> dict[str: np.ndarray]:

        # This will only get the threshold, i.e. the only parameter for the daily index
        parname = 'threshold'

        thr_data = self._parameters[parname].get_data(time, **case.tags)
        return {'threshold' :thr_data.values.squeeze()}

    def get_index_pooled(self, time: ts.TimeStep, case) -> tuple[dict[str: np.ndarray], dict]:

        index_pooled = {}
        metadata     = {}
        for key, varname in self.pooled_vars.items():
            if not self._index_pooled.check_data(time, **case.tags, **{self.pooled_var_name: varname}):
                return {k: None for k in self.pooled_vars.keys()}, {}
            
            this_index = self._index_pooled.get_data(time, **case.tags, **{self.pooled_var_name: varname})
            index_pooled[key] = this_index.values.squeeze()

            for key in self.options['propagate_metadata']:
                if key in this_index.attrs:
                    if key not in metadata:
                        metadata[key] = [this_index.attrs[key]]
                    else:
                        metadata[key].append(this_index.attrs[key])
        
        metadata = {k:','.join(v) for k,v in metadata.items()}
        metadata.update(case.options.copy())

        return index_pooled, metadata

    def get_last_ts_index(self, **kwargs) -> ts.TimeStep:
        index_pooled_cases = self.cases_pooling[-1]
        last_ts_pooled_index = None
        for case in index_pooled_cases.values():
            for var in self.pooled_vars.values():
                now = kwargs.pop('now', None) if last_ts_pooled_index is None else last_ts_pooled_index.end + timedelta(days = 1)
                index = self._index_pooled.get_last_ts(now = now, **case.tags, **{self.pooled_var_name:var}, **kwargs)
                if index is not None:
                    last_ts_pooled_index = index if last_ts_pooled_index is None else min(index, last_ts_pooled_index)
                else:
                    last_ts_pooled_index = None
                    break
        
        return last_ts_pooled_index

    def get_last_ts_index_daily(self, **kwargs) -> ts.TimeStep:
        index_daily_cases = self.cases[-1]
        last_ts_daily_index = None
        for case in index_daily_cases.values():
            now = kwargs.pop('now', None) if last_ts_daily_index is None else last_ts_daily_index.end + timedelta(days = 1)
            index = self._index.get_last_ts(now = now, **case.tags, **kwargs)
            if index is not None:
                last_ts_daily_index = index if last_ts_daily_index is None else min(index, last_ts_daily_index)
            else:
                last_ts_daily_index = None
                break
        
        return last_ts_daily_index

    def get_last_ts_available(self, **kwargs) -> ts.TimeStep:
        last_ts_daily_index = self.get_last_ts_index_daily(**kwargs)
        last_ts_data = super().get_last_ts_available(**kwargs)
        
        # Get the largest non-None value
        last_ts_available = max(
            filter(None, [last_ts_daily_index, last_ts_data]),
            default=None
        )
        
        return last_ts_available

class LFI(DRYESThrBasedIndex):
    index_name = 'LFI'

    # default options
    default_options = {
        'thr_quantile' :  0.05,  # quantile for the threshold calculation
        'thr_window'   : 31,     # window size for the threshold calculation
        'min_duration' :  5,     # minimum duration of a spell
        'min_interval' : 10,     # minimum interval between spells
        'min_nevents'  :  5,     # minimum number of events in the historic period to calculate lambda (LFI only)

        'look_ahead'        : False,
    }

    option_cases_normalising = {
        'parameters_lambda' : ['min_nevents', 'min_duration', 'min_interval']
    }

    direction = -1

    parameters             = ['threshold', 'lambda']
    parameters_normalising = ['lambda']

    def _check_io_index(self, io_options, update_existing=False):
        
        if not hasattr(self, '_index_norm') or update_existing:
            # check that we have an output specification for the index
            if 'index' not in io_options: raise ValueError('No output path specified for the index.')
            self._index_norm = io_options['index']
            self._index_norm._template = self.output_template        

        if not hasattr(self, '_index_pooled') or update_existing:
            # check that we have an output specification for the pooled index
            self._index_pooled = io_options.get('index_pooled')
            if self._index_pooled is None:
                from d3tools.data.memory_dataset import MemoryDataset
                key_pattern = self._index_norm.key_pattern.replace('.tif', '_pooled.tif')
                self._index_pooled = MemoryDataset(key_pattern)
                if self._index_pooled.has_tiles: self._index_pooled.tile_names = self._data.tile_names
            
            self._index_pooled._template = self.output_template

        if not hasattr(self, '_index') or update_existing:
            # check that we have an output specification for the daily components of the index
            self._index = io_options.get('index_daily')
            if self._index is None:
                from d3tools.data.memory_dataset import MemoryDataset
                key_pattern = self._index_norm.key_pattern.replace('.tif', '_daily.tif')
                self._index = MemoryDataset(key_pattern)
                if self._index_norm.has_tiles: self._index_norm.tile_names = self._data.tile_names
            
            self._index._template = self.output_template

    def _check_io_parameters(self, io_options: dict, update_existing = False) -> None:
        super()._check_io_parameters(io_options, update_existing)
        for par in self.parameters_normalising:
            if par not in io_options: raise ValueError(f'No source/destination specified for parameter {par}.')
            self._parameters[par] = io_options[par]
            self._parameters[par]._template = self.output_template

    def _set_cases(self) -> None:
        self.cases, self.cases_pooling, self.cases_normalising = self._get_cases()

    def _get_cases(self) -> dict:
        cases, cases_pooling = super()._get_cases()
        options = self.options
        
        last_options  = cases_pooling.options[-1]
        cases_normalising = CaseManager(last_options, 'index_pooled')

        for layer, opts in self.option_cases_normalising.items():
            these_options = {k:options.get(k, self.default_options.get(k)) for k in opts}
            cases_normalising.add_layer(these_options, layer)

        return cases, cases_pooling, cases_normalising

    def _make_other_parameters(self, history: ts.TimeRange) -> None:
        # to calculate the lambda:
        # 1. calculate the daily index for the history period -> this will do it for all the cases
        DRYESIndex._make_index(self, history, history, 'd')

        # 2. pool the data for the history period, also for all the cases
        super()._make_index_pooled(history, history, 'd')

        # 3. get the lambdas for each normalising case
        #    - loop trhought the pooling cases
        for pooled_case_id, pool_case in self.cases_normalising['index_pooled'].items():

            # loop through the days in the history period (skip the first day and set it as previous)
            days = history.days
            pooled_index, _ = self.get_index_pooled(days[0], pool_case)
            sduration_prev = pooled_index['sduration']
            sintensity_prev = pooled_index['sintensity']

            for i, day in enumerate(days[1:]):
                pooled_index, _ = self.get_index_pooled(day, pool_case)
                sduration = pooled_index['sduration']
                sintensity = pooled_index['sintensity']
                pool_cases = pooled_index['case']

                # get where the spell ends (pool_case == 6), and when the spell was long enough to be an event
                spell_ends = pool_cases == 6
                event_ends = np.logical_and(spell_ends, sduration_prev >= pool_case.options['min_duration'])

                if i == 0:
                    # first day, we need to initialise the cumulative intensity and number of events
                    cum_intensity = np.zeros_like(pool_cases)
                    n_events      = np.zeros_like(pool_cases)

                # get the cumulative intensity and number of events for this day
                cum_intensity[event_ends] += sintensity_prev[event_ends]
                n_events      += event_ends

                # set the previous values for the next day
                sduration_prev = sduration
                sintensity_prev = sintensity

            # now loop through the normalising cases that are under this pooling case
            for lambda_case_id, lambda_case in self.cases_normalising['parameters_lambda'].items():
                if not lambda_case_id.startswith(pooled_case_id):
                    continue
                    
                # filter to only calculate the lambda where we have enough events
                cum_intensity = np.where(n_events >= lambda_case.options['min_nevents'], cum_intensity, np.nan)
                n_events      = np.where(n_events >= lambda_case.options['min_nevents'], n_events, 1e-6)

                # calculate the lambda (lambda = 1/(avg_intensity) = 1/(cum_intensity/n_events))
                avg_intensity = cum_intensity / n_events
                avg_intensity = np.where(avg_intensity <1e-6, 1e-6 , avg_intensity)
                this_lambda = 1 / avg_intensity
                this_lambda = np.where(avg_intensity == 1e-6, np.nan, this_lambda)

                # save the lambda
                metadata = lambda_case.options.copy()
                metadata.update({'reference': f'{history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}'})
                tags = lambda_case.tags
                self._parameters['lambda'].write_data(this_lambda, metadata = metadata, **tags)

    def _make_index(self, current: ts.TimeRange, reference: ts.TimeRange, frequency: str) -> None:

        ## figure out the last daily index, to adjust current if needed
        last_pooled  = super().get_last_ts_index(now = current.end, lim = current.start)

        # make the daily and pooled index from the superclass if we have to
        if last_pooled is None: last_pooled = ts.Day.from_date(current.start) - 1
        if last_pooled.end < current.end:
            pool_tr = ts.TimeRange(last_pooled.start + timedelta(days = 1), current.end)
            super()._make_index(pool_tr, reference, frequency)

        # make the normalisation from here!
        for pooled_case_id, pooled_case in self.cases_normalising['index_pooled'].items():
            # get the timesteps for which we need to calculate the index
            timesteps:list[ts.TimeStep] = current.get_timesteps(frequency)

            # now loop through the normalising cases that are under this pooling case
            for lambda_case_id, lambda_case in self.cases_normalising['parameters_lambda'].items():
                if not lambda_case_id.startswith(pooled_case_id):
                    continue

                # get the parameters for this case (lambda)
                lambda_data = self._parameters['lambda'].get_data(**lambda_case.tags)

                # loop through all the timesteps
                for time in timesteps:

                    # get the daily index for this time
                    this_pooled_index, metadata = self.get_index_pooled(time, pooled_case)
                    intensity = this_pooled_index['intensity'] # we really only care about the intensity here
                    if intensity is None:
                        continue

                    # normalise the intensity (Poisson distribution)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        normal_intensity = 1 - np.exp(-lambda_data * intensity)
                    normal_intensity = np.where(intensity <= 0, 0, normal_intensity)
                    normal_intensity = np.where(np.isnan(lambda_data), np.nan, normal_intensity)

                    # save the data
                    metadata.update(lambda_case.options.copy())
                    metadata.update({'reference': f'{reference.start:%d/%m/%Y}-{reference.end:%d/%m/%Y}'})
                    tags = lambda_case.tags
                    self._index_norm.write_data(normal_intensity, time = time, metadata = metadata, **tags)

    def get_last_ts_index(self, **kwargs) -> ts.TimeStep:
        index_norm_cases = self.self.cases_normalising[-1]
        last_ts_norm_index = None
        for case in index_norm_cases.values():
            now = kwargs.pop('now', None) if last_ts_norm_index is None else last_ts_norm_index.end + timedelta(days = 1)
            index = self._index_norm.get_last_ts(now = now, **case.tags, **kwargs)
            if index is not None:
                last_ts_norm_index = index if last_ts_norm_index is None else min(index, last_ts_norm_index)
            else:
                last_ts_norm_index = None
                break
        
        return last_ts_norm_index

    def get_last_ts_available(self, **kwargs) -> ts.TimeStep:
        last_ts_pooled_index = super().get_last_ts_index(**kwargs)
        last_ts_daily_index  = self.get_last_ts_index_daily(**kwargs)
        last_ts_data         = super().get_last_ts_available(**kwargs)
        
        # Get the largest non-None value
        last_ts_available = max(
            filter(None, [last_ts_pooled_index, last_ts_daily_index, last_ts_data]),
            default=None
        )
        
        return last_ts_available

class HCWI(DRYESThrBasedIndex):
    index_name = 'HCWI'
    default_options = {
        'thr_window'   : 11,    # window size for the threshold calculation
        'min_interval' :  1,    # minimum interval between spells
        'min_duration' :  3,    # minimum duration of a spell

        # options to set the tags for the input data
        'Ttypes'       : {'max' : 'max', 'min' : 'min'}, # the types of temperature data to use
        'Ttype_name'   : 'Ttype', # the name of the Ttype variable in the data
        
        'daily_var_name' : 'dvar',
        'daily_vars'     : {'dintensity' : 'dintensity', 'ishit' : 'ishit'},
    }

    def _set_io_tags(self) -> None:
        self.Ttype_name = self.options.pop('Ttype_name')
        self.Ttypes     = self.options.pop('Ttypes')

        self.daily_var_name = self.options.pop('daily_var_name')
        self.daily_vars     = self.options.pop('daily_vars')

        super()._set_io_tags()

    def _check_io_data(self, io_options: dict) -> None:
        self._set_io_tags()
        data_ds = io_options['data']

        if not len(self.Ttypes) == 2 or not all(['min' in self.Ttypes.keys(), 'max' in self.Ttypes.keys()]):
            raise ValueError('Ttypes must be a dictionary with two elements: min and max.')

        self._raw_inputs = {'min': data_ds.update(**{self.Ttype_name: self.Ttypes['min']}), 
                            'max': data_ds.update(**{self.Ttype_name: self.Ttypes['max']})}

        # ensure we have the same templates for everything
        self._raw_inputs['max']._template = self._raw_inputs['min']._template
        self.output_template = self._raw_inputs['min']._template

        from d3tools.data.memory_dataset import MemoryDataset
        key_pattern = self._raw_inputs['min'].key_pattern.replace('min', '').replace('.tif', '.nc')
        self._data = MemoryDataset(key_pattern)
        if self._data.has_tiles: self._data.tile_names = self._raw_inputs['min'].tile_names

        self._data.set_parents({'min': self._raw_inputs['min'], 'max': self._raw_inputs['max']},
                               lambda min, max: xr.Dataset({'min': min, 'max': max}))

    def _make_thresholds(self, history, data_case, data_case_id):
        for Ttype, var in self._raw_inputs.items():
            super()._make_thresholds(history, data_case, data_case_id, var = var, var_tags = {self.Ttype_name: self.Ttypes[Ttype]})

    def calc_index(self,
                   data: np.ndarray, parameters: dict[str, np.ndarray],
                   options: dict, step = 1, **kwargs) -> list[tuple[np.ndarray, dict]]:

        thresholds = parameters['threshold']
        dindex = calc_dintensity(data, thresholds, self.direction, get_hits=True)

        output = [(value, {self.daily_var_name: self.daily_vars[key]}) for key, value in dindex.items()]

        return output

    def get_data(self, time: ts.TimeStep, case) -> tuple[np.ndarray, dict]:

        # the output here should be 3d with the first dimension being the Ttype in the order [min, max]
        if not self._data.check_data(time, **case.options):
            return None

        data_ds = self._data.get_data(time, **case.options)
        data_da = data_ds.to_array(self.Ttype_name)   

        metadata = {}
        for key in self.options['propagate_metadata']:
            for var in data_ds.data_vars:
                data = data_ds[var]
                if key in data.attrs:
                    if key in metadata:
                        metadata[key].append(data.attrs[key])
                    else:
                        metadata[key] = [data.attrs[key]]
            metadata = {k: ','.join(v) for k, v in metadata.items()}

        return data_da.values.squeeze(), metadata

    def get_parameters(self, time: ts.TimeStep, case) -> dict[str: np.ndarray]:

        # the output here should be 3d with the first dimension being the Ttype in the order [min, max]
        parname = 'threshold'

        thr_data_min = self._parameters[parname].get_data(time, **case.tags, **{self.Ttype_name: self.Ttypes['min']})
        thr_data_max = self._parameters[parname].get_data(time, **case.tags, **{self.Ttype_name: self.Ttypes['max']})

        thr_ds = xr.Dataset({'min': thr_data_min, 'max': thr_data_max})
        thr_da = thr_ds.to_array(self.Ttype_name)

        return {'threshold' :thr_da.values.squeeze()}

    def get_index_daily(self, time: ts.TimeRange, case) -> tuple[dict[str:np.ndarray], dict]:

        if not isinstance(time, ts.Day):
            return super().get_index_daily(time, case)

        if not self._index.check_data(time, **case.tags, **{self.daily_var_name: self.daily_vars['dintensity']}):
            return {'dintensity' : None, 'ishit' : None}, {}
        elif not self._index.check_data(time, **case.tags, **{self.daily_var_name: self.daily_vars['ishit']}):
            return {'dintensity' : None, 'ishit' : None}, {}
        
        dintensity = self._index.get_data(time, **case.tags, **{self.daily_var_name: self.daily_vars['dintensity']})
        ishit = self._index.get_data(time, **case.tags, **{self.daily_var_name: self.daily_vars['ishit']}).astype('int8')

        metadata = {}
        for key in self.options['propagate_metadata']:
            for data in [dintensity, ishit]:
                if key in data.attrs:
                    if key in metadata:
                        metadata[key].append(data.attrs[key])
                    else:
                        metadata[key] = [data.attrs[key]]

        metadata = {k: ','.join(v) for k, v in metadata.items()}
        metadata.update(case.options.copy())

        return {'dintensity' : dintensity.values.squeeze(), 'ishit' : ishit.values.squeeze()}, metadata

    def get_last_ts_index_daily(self, **kwargs) -> ts.TimeStep:
        index_daily_cases = self.cases[-1]
        last_ts_daily_index = None
        for case in index_daily_cases.values():
            for var in self.daily_vars.values():
                now = kwargs.pop('now', None) if last_ts_daily_index is None else last_ts_daily_index.end + timedelta(days = 1)
                index = self._index.get_last_ts(now = now, **case.tags, **{self.daily_var_name:var}, **kwargs)
                if index is not None:
                    last_ts_daily_index = index if last_ts_daily_index is None else min(index, last_ts_daily_index)
                else:
                    last_ts_daily_index = None
                    break
        
        return last_ts_daily_index

class HWI(HCWI):
    index_name = 'HWI'
    direction = 1

    default_options = {
        'thr_quantile' : 0.9, # quantile for the threshold calculation
    }

class CWI(HCWI):
    index_name = 'CWI'
    direction = -1

    default_options = {
        'thr_quantile' : 0.1, # quantile for the threshold calculation
    }