from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import tempfile
import subprocess
import shutil
import copy

from typing import Generator

from .dryes_index import DRYESIndex

import d3tools.timestepping as ts
from d3tools.cases import CaseManager

from ..utils.parse import make_case_hierarchy, options_to_cases

class DRYESThrBasedIndex(DRYESIndex):
    """
    This class implements threshold based indices.
    The indices that are relevant for this class are: LFI (Low Flow Index) and HCWI (Heat and Cold Wave Index).
    """

    index_name = 'Threshold based index'

    # default options
    default_options = {
        'thr_quantile' :  0.1,   # quantile for the threshold calculation
        'thr_window'   :  1,     # window size for the threshold calculation
        'min_interval' :  1,     # minimum interval between spells
        #'min_duration' :  5,     # minimum duration of a spell
        
        'look_ahead'       : False,   # if True, it looks min_interval days ahead to see if the spell continues
        'count_with_pools' : False,   # if True, the pool days are added to the duration and intensity of the spell

        'cdo_path'     :  '/usr/bin/cdo', # path to the cdo executable

        # options to set the tags of the output data
        'pooled_var_name': 'var',
        'pooled_vars'    : {'intensity' : 'intensity', 'duration' : 'duration', 'interval' : 'interval'},
    }

    option_cases = {
        'parameters_thr' : ['thr_quantile', 'thr_window', 'cdo_path'],
        'index_daily'    : []
    }

    option_cases_pooling = {
        'parameters'   : [],
        'index_pooled' : ['min_interval', 'look_ahead', 'count_with_pools'],
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
            self.max_look_ahead = min_interval
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
            if len(other_par) == 0:
                continue

            # loop through the cases for the thresholds
            thr_cases = self.cases['parameters_thr']
            for thr_case_id, thr_case in thr_cases.items():
                if thr_case_id.startswith(data_case_id):
                    self._make_other_parameters(history, data_case, thr_case)

    def _make_thresholds(self, history: ts.TimeRange, data_case: dict, data_case_id: str, var = None, var_tags = None) -> None:
        
        var      = var      or self._data
        var_tags = var_tags or {}

        days = history.days
        
        data_case.options.update(var_tags)
        datasets = [var.get_data(day, as_is=True, **data_case.options).squeeze().expand_dims('time').assign_coords(time=[day.start]) for day in days]
        combined = xr.concat(datasets, dim='time')

        var.set_template(datasets[0])

        tmpdir = tempfile.mkdtemp()

        # save the data to a temporary file as netcdf with a time dimension
        data_nc = f'{tmpdir}/data.nc'

        # Specify encoding to ensure correct data types
        import netCDF4
        combined.to_netcdf(data_nc, format = 'NETCDF4', engine = 'netcdf4')

        # get the timesteps for which we need to calculate the parameters - thresholds are always daily (incl. 29th Feb)
        timesteps:list[ts.TimeStep] = ts.TimeRange('1904-01-01', '1904-12-31').days

        # loop through the cases for the thresholds
        thr_cases = self.cases['parameters_thr']
        for thr_case_id, thr_case in thr_cases.items():
            if not thr_case_id.startswith(data_case_id):
                continue

            thresholds = calc_thresholds_cdo(data_nc, history, thr_case)

            for time, thr_data in zip(timesteps, thresholds):
                metadata = thr_case.options.copy()
                metadata.update({'reference': f'{history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}'})
                tags = thr_case.tags | var_tags
                self._parameters['threshold'].write_data(thr_data, time = time, time_format = '%d/%m', metadata = metadata, **tags)

        shutil.rmtree(tmpdir)

    def _make_index(self, current: ts.TimeRange, reference: ts.TimeRange, frequency: str) -> None:

        # adjust the current considering we might need some look-ahead time
        extended_current = current.extend(ts.TimeWindow(self.max_look_ahead, 'd'))

        ## figure out the last pooled and last daily index, to adjust current if needed
        last_daily  = super().get_last_ts(now = extended_current.end, lim = current.start)
        last_pooled    = self.get_last_ts(now = last_daily.end if last_daily is not None else current.end,
                                          lim = current.start)

        # make the daily index here:
        if last_daily is None: last_daily = ts.Day.from_date(current.start) - 1
        if last_daily.end < extended_current.end:
            daily_tr = ts.TimeRange(last_daily.start + timedelta(days = 1), extended_current.end)
            super()._make_index(daily_tr, reference, 'd')

        # and then pool it!
        if last_pooled is None: last_pooled = ts.Day.from_date(current.start) - 1
        if last_pooled.end < extended_current.end:
            pooled_tr = ts.TimeRange(last_pooled.start + timedelta(days = 1), current.end)
            self._make_index_pooled(pooled_tr, reference, frequency)

    def _make_index_pooled(self, current: ts.TimeRange, reference: ts.TimeRange, frequency: str) -> None:
        
        for index_daily_case_id, index_daily_case in self.cases_pooling['index_daily'].items():

            data_ts_unit = self._data.estimate_timestep(**index_daily_case.options).unit
            if frequency is not None:
                if not ts.unit_is_multiple(frequency, data_ts_unit):
                    raise ValueError(f'The data timestep unit ({data_ts_unit}) is not a multiple of the frequency requeested ({frequency}).')
            else:
                frequency = data_ts_unit

            # get the timesteps for which we need to calculate the index
            timesteps:list[ts.TimeStep] = current.get_timesteps(frequency)

            # get the cases for the last layer of the parameters
            parameter_layers = [l for l in self.cases_pooling._lyrmap if l.startswith('parameters')]
            this_data_par_cases = {id:case for id,case in self.cases_pooling[parameter_layers[-1]].items() if id.startswith(index_daily_case_id)}

            # loop through all the timesteps
            for time in timesteps:

                # get the daily index for this time
                this_daily_index = self.get_index_daily(ts.TimeRange(time.start, time.end), index_daily_case)

                # the eventual daily index of the look-ahead period
                if self.max_look_ahead == 0:
                    future_daily_index = None
                else:
                    lookahead_period = ts.TimeRange(time.end + timedelta(1), time.end + timedelta(days = self.max_look_ahead))
                    future_daily_index = self.get_index_daily(lookahead_period, index_daily_case)
                
                # loop through all parameter layers for this data case
                for par_case_id, par_case in this_data_par_cases.items():
                    # get the parameters for this case
                    parameters_np = self.get_parameters_pooling(time, par_case)

                    # loop through all index layers for this parameter case
                    input = (this_daily_index, future_daily_index)
                    for idx_case_id, idx_case in self.cases_pooling[-1].items():
                        if not idx_case_id.startswith(par_case_id):
                            continue
                        
                        previous_index = self.get_index_pooled(time - 1, idx_case)

                        this_input = copy.deepcopy(input)
                        this_index = self.pool_index(this_input, parameters_np, previous_index, idx_case.options)

                        metadata = idx_case.options.copy()
                        metadata.update({'reference': f'{reference.start:%d/%m/%Y}-{reference.end:%d/%m/%Y}'})
                        tags = idx_case.tags

                        for index, other_tags in this_index:
                            self._index_pooled.write_data(index, time = time, metadata = metadata, **tags, **other_tags)

    def calc_index(self,
                   data: np.ndarray, parameters: dict[str, np.ndarray],
                   options: dict, step = 1, **kwargs) -> np.ndarray:

        thresholds = parameters['threshold']
        return (data - thresholds) * self.direction
    
    def pool_index(self,
                   daily_index: tuple[tuple[np.ndarray]], parameters: dict[str:np.ndarray], current_index: tuple[np.ndarray],
                   options: dict, **kwargs) -> list[tuple[np.ndarray:dict]]:
        
        current_daily_index, future_daily_index = daily_index
        intensity, duration, interval = current_index

        daily_intensity = current_daily_index[0]
        daily_spell     = current_daily_index[1]
        n = daily_intensity.shape[0]

        if options['look_ahead']:
            future_intensity = future_daily_index[0][:options['min_interval']]
            future_spell     = future_daily_index[1][:options['min_interval']]
            daily_intensity = np.concatenate((daily_intensity, future_intensity))
            daily_spell     = np.concatenate((daily_spell, future_spell))

        if intensity is None:
            intensity = np.zeros_like(current_daily_index[0][0])
        if duration is None:
            duration = np.zeros_like(current_daily_index[0][0], dtype = int)
        if interval is None:
            interval = np.zeros_like(current_daily_index[0][0], dtype = int) + int(options['min_interval']) + 1

        for i in range(n):
            # where we have a spell, add the intensity and increase the duration by one, set interval since last spell to 0
            is_spell = daily_spell[i] == 1
            intensity[is_spell] += daily_intensity[i][is_spell]
            duration[is_spell]  += 1
            interval[is_spell]   = 0

            # where we have the end of a spell, we need to check if this is a pool day or not
            is_end_of_spell = np.logical_and(daily_spell[i] == 0, duration > 0)

            # first let's check when the next spell starts, if we have to look ahead
            xx,yy = np.where(is_end_of_spell)
            if options['look_ahead']:
                # future_interval = np.argmax(daily_spell[i+1:] > 0, axis=0)
                # future_interval[future_interval == 0] = options['min_interval'] + 1
                # future_interval = np.where(daily_spell[i+1:].any(axis=0), future_interval, options['min_interval'] + 1)
                future_interval = np.zeros_like(daily_spell[i])
                for x, y in zip(xx, yy):
                    next_spell_idx = np.argmax(daily_spell[i+1:, x, y] > 0)
                    if daily_spell[i+1:, x, y].any():
                        future_interval[x, y] = next_spell_idx
                    else:
                        future_interval[x, y] = options['min_interval'] +1
            else:
                future_interval = np.zeros_like(daily_spell[i])

            # pool days are where the interval (past + eventually, future) is less than the minimum interval
            is_pool_day = np.logical_and(is_end_of_spell, interval + future_interval <= options['min_interval'])

            # where we have a pool day, increase the interval
            interval[is_pool_day] += 1
            # and depending on the option, add the intensity to the previous spell
            if options['count_with_pools']:
                intensity[is_pool_day] += daily_intensity[i][is_pool_day]
                duration[is_pool_day]  += 1

            # review is_end_of_spell to remove the pool days
            is_end_of_spell = np.logical_and(is_end_of_spell, ~is_pool_day)

            # where we have the end of a spell, set intensity and duration to 0, and interval to 1
            interval[is_end_of_spell] = 1
            intensity[is_end_of_spell] = 0
            duration[is_end_of_spell] = 0

            # everywhere else, increase the interval
            interval[~np.logical_or.reduce([is_end_of_spell, is_pool_day, is_spell])] += 1

            output = [(intensity, {self.pooled_var_name: self.pooled_vars['intensity']}),
                      (duration,  {self.pooled_var_name: self.pooled_vars['duration']}),
                      (interval,  {self.pooled_var_name: self.pooled_vars['interval']})]
            
            return output

    def get_index_daily(self, time: ts.TimeRange, case) -> np.ndarray:

        if not isinstance(time, ts.Day):
            days_in_time = time.days
            all_dintensity_np = []
            all_sbool_np      = []
            for t in days_in_time:
                this_dintensity_np, this_sbool_np = self.get_index_daily(t, case)
                if this_dintensity_np is None or this_sbool_np is None:
                    continue ##TODO: ADD A WARNING OR SOMETHING
                all_dintensity_np.append(this_dintensity_np)
                all_sbool_np.append(this_sbool_np)
            
            all_dintensity_np = np.stack(all_dintensity_np)
            all_sbool_np      = np.stack(all_sbool_np)
            return all_dintensity_np, all_sbool_np

        if not self._index.check_data(time, **case.tags):
            return None, None

        dintensity = self._index.get_data(time, **case.tags)
        sbool      = xr.where(dintensity > 0, 1, 0)
        
        return dintensity.values.squeeze(), sbool.values.squeeze()

    def get_parameters_pooling(self, time: datetime, case) -> dict[str, np.ndarray]:
        parameters_xr = {parname: self._parameters[parname].get_data(time, **case.tags) for parname in self.parameters_pooling}
        parameters_np = {parname: par.values.squeeze() for parname, par in parameters_xr.items()}
        return parameters_np

    def get_index_pooled(self, time: ts.TimeStep, case) -> np.ndarray:

        if not self._index_pooled.check_data(time, **case.tags, **{self.pooled_var_name: self.pooled_vars['intensity']}):
            return None, None, None
        elif not self._index_pooled.check_data(time, **case.tags, **{self.pooled_var_name: self.pooled_vars['duration']}):
            return None, None, None
        elif not self._index_pooled.check_data(time, **case.tags, **{self.pooled_var_name: self.pooled_vars['interval']}):
            return None, None, None
        
        intensity = self._index_pooled.get_data(time, **case.tags, **{self.pooled_var_name: self.pooled_vars['intensity']})
        duration = self._index_pooled.get_data(time, **case.tags, **{self.pooled_var_name: self.pooled_vars['duration']})
        interval = self._index_pooled.get_data(time, **case.tags, **{self.pooled_var_name: self.pooled_vars['interval']})
        
        return intensity.values.squeeze(), duration.values.squeeze(), interval.values.squeeze()

    def get_last_ts(self, inputs = False, **kwargs) -> ts.TimeStep:
        index_pooled_cases = self.cases_pooling[-1]
        last_ts_pooled_index = None
        for case in index_pooled_cases.values():
            now = kwargs.pop('now', None) if last_ts_pooled_index is None else last_ts_pooled_index.end + timedelta(days = 1)
            index = self._index_pooled.get_last_ts(now = now, **case.tags, **kwargs)
            if index is not None:
                last_ts_pooled_index = index if last_ts_pooled_index is None else min(index, last_ts_pooled_index)
            else:
                last_ts_pooled_index = None
                break
        
        if not inputs:
            return last_ts_pooled_index
        
        last_ts_index, other = super().get_last_ts(inputs = True, **kwargs)
        other['index_daily'] = last_ts_index
        return last_ts_pooled_index, other

class LFI(DRYESThrBasedIndex):
    index_name = 'LFI'

    # default options
    default_options = {
        'thr_quantile' :  0.05,  # quantile for the threshold calculation
        'thr_window'   : 31,     # window size for the threshold calculation
        'min_duration' :  5,     # minimum duration of a spell
        'min_interval' : 10,     # minimum interval between spells
        'min_nevents'  :  5,     # minimum number of events in the historic period to calculate lambda (LFI only)
    }

    option_cases = {
        'parameters_thr' : ['thr_quantile', 'thr_window', 'cdo_path'],
        'index_daily'    : []
    }

    option_cases_normalising = {
        'parameters_lambda' : ['min_nevents', 'min_duration', 'min_interval']
    }

    direction = -1

    parameters             = ['threshold']
    parameters_normalising = ['lambda']

    def _check_io_parameters(self, io_options: dict, update_existing = False) -> None:
        super()._check_io_parameters(io_options, update_existing)
        for par in self.parameters_normalising:
            if par not in io_options: raise ValueError(f'No source/destination specified for parameter {par}.')
            self._parameters[par] = io_options[par]
            self._parameters[par]._template = self.output_template
    
    def _make_other_parameters(self, history: ts.TimeRange, data_case: dict, thr_case: dict) -> None:
        pass

class HCWI(DRYESThrBasedIndex):
    index_name = 'HCWI'
    default_options = {
        'thr_window'   : 11,    # window size for the threshold calculation
        'min_interval' :  1,    # minimum interval between spells
        #'min_duration' :  3,    # minimum duration of a spell

        # options to set the tags for the input data
        'Ttypes'       : {'max' : 'max', 'min' : 'min'}, # the types of temperature data to use
        'Ttype_name'   : 'Ttype', # the name of the Ttype variable in the data
        
        'daily_var_name' : 'dvar',
        'daily_vars'     : {'dintensity' : 'dintensity', 'sbool' : 'sbool'},
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
 
        self._data.set_parents({'min': self._raw_inputs['min'], 'max': self._raw_inputs['max']},
                               lambda min, max: xr.Dataset({'min': min, 'max': max}))

    def _make_thresholds(self, history, data_case, data_case_id):
        for Ttype, var in self._raw_inputs.items():
            super()._make_thresholds(history, data_case, data_case_id, var = var, var_tags = {self.Ttype_name: self.Ttypes[Ttype]})

    def calc_index(self,
                   data: np.ndarray, parameters: dict[str, np.ndarray],
                   options: dict, step = 1, **kwargs) -> list[tuple[np.ndarray, dict]]:

        both_deviations = super().calc_index(data, parameters, options, step, **kwargs)

        sbool       = np.logical_and(both_deviations[0] >= 0, both_deviations[1] >= 0) * 1
        dintensity  = np.mean(np.where(both_deviations>0, both_deviations, 0), axis = 0)

        return [(dintensity, {self.daily_var_name: self.daily_vars['dintensity']}),
                (sbool,      {self.daily_var_name: self.daily_vars['sbool']})]

    def get_data(self, time: ts.TimeStep, case) -> np.ndarray:

        # the output here should be 3d with the first dimension being the Ttype in the order [min, max]

        data_ds = self._data.get_data(time, case)
        data_da = data_ds.to_array(self.Ttype_name)   

        return data_da.values.squeeze()

    def get_parameters(self, time: ts.TimeStep, case) -> dict[str: np.ndarray]:

        # the output here should be 3d with the first dimension being the Ttype in the order [min, max]
        parname = 'threshold'

        thr_data_min = self._parameters[parname].get_data(time, **case.tags, **{self.Ttype_name: self.Ttypes['min']})
        thr_data_max = self._parameters[parname].get_data(time, **case.tags, **{self.Ttype_name: self.Ttypes['max']})

        thr_ds = xr.Dataset({'min': thr_data_min, 'max': thr_data_max})
        thr_da = thr_ds.to_array(self.Ttype_name)

        return {'threshold' :thr_da.values.squeeze()}

    def get_index_daily(self, time: ts.TimeRange, case) -> np.ndarray:

        if not isinstance(time, ts.Day):
            return super().get_index_daily(time, case)

        if not self._index.check_data(time, **case.tags, **{self.daily_var_name: self.daily_vars['dintensity']}):
            return None, None
        elif not self._index.check_data(time, **case.tags, **{self.daily_var_name: self.daily_vars['sbool']}):
            return None, None
        
        dintensity = self._index.get_data(time, **case.tags, **{self.daily_var_name: self.daily_vars['dintensity']})
        sbool = self._index.get_data(time, **case.tags, **{self.daily_var_name: self.daily_vars['sbool']})
        
        return dintensity.values.squeeze(), sbool.values.squeeze()
        
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

def calc_thresholds_cdo(data_nc: xr.DataArray,
                        history: ts.TimeRange,
                        case) -> Generator[tuple[xr.DataArray, dict], None, None]:
    """
    This will only do the threshold calculation using CDO.
    """
    import os

    tmpdir = os.path.dirname(data_nc)

    # set the number of bins in CDO as an environment variable
    window_size = case.options['thr_window']
    history_start = history.start.year
    history_end = history.end.year
    
    CDO_PCTL_NBINS= window_size * (history_end - history_start + 1) * 2 + 2
    os.environ['CDO_NUMBINS'] = str(CDO_PCTL_NBINS)

    # calculate running max and min of data using CDO
    datamin_nc = f'{tmpdir}/datamin.nc'
    datamax_nc = f'{tmpdir}/datamax.nc'

    cdo_path = case.options['cdo_path']
    cdo_cmd =  f'{cdo_path} ydrunmin,{window_size},rm=c {data_nc} {datamin_nc}'
    subprocess.run(cdo_cmd, shell = True)

    cdo_cmd = f'{cdo_path} ydrunmax,{window_size},rm=c {data_nc} {datamax_nc}'
    subprocess.run(cdo_cmd, shell = True)

    # calculate the thresholds
    threshold_quantile = int(case.options['thr_quantile']*100)
    threshold_nc = f'{tmpdir}/threshold.nc'

    cdo_cmd = f'{cdo_path} ydrunpctl,{threshold_quantile},{window_size},rm=c,pm=r8 {data_nc} {datamin_nc} {datamax_nc} {threshold_nc}'
    subprocess.run(cdo_cmd, shell = True)

    # read the threshold data
    thresholds = xr.open_dataset(threshold_nc)
    
    # get the crs and write it to the threshold data
    thresholds_da = thresholds['__xarray_dataarray_variable__']
    # crs = combined.rio.crs
    # thresholds_da = thresholds['__xarray_dataarray_variable__'].rio.write_crs(crs)

    # add a 1-d "band" dimension
    thresholds_da = thresholds_da.expand_dims('band')
    
    # loop over the timesteps and yield the threshold
    days = thresholds_da.time.values
    for i, date in enumerate(days):
        time = datetime.fromtimestamp(date.astype('O') / 1e9)
        if time.month == 2 and time.day == 29:
            # do the average between the thresholds for 28th of Feb and 1st of Mar
            threshold_data = (thresholds_da.isel(time = i-1) + thresholds_da.isel(time = i+1)) / 2
        else:
            threshold_data = thresholds_da.isel(time = i).drop_vars('time')

        yield threshold_data