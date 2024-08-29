from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import warnings
import tempfile
import subprocess
import shutil

from typing import Optional, Generator, Iterable

from .dryes_index import DRYESIndex

from ..tools.timestepping import TimeRange, Day
from ..tools.timestepping.fixed_num_timestep import FixedNTimeStep
from ..tools.timestepping.timestep import TimeStep

from ..tools.data import Dataset
from ..utils.parse import make_case_hierarchy, options_to_cases

class DRYESThrBasedIndex(DRYESIndex):
    """
    This class implements threshold based indices.
    The indices that are relevant for this class are: LFI (Low Flow Index) and HCWI (Heat and Cold Wave Index).
    """

    index_name = 'Threshold based index'
    # this flag is used to determine if we need to process all timesteps continuously
    # or if we can have gaps in the processing
    # Threshold based indices are always continuous as they require pooling (for other indices, it depends on the time aggregation)
    iscontinuous = True

    # default options
    default_options = {
        'thr_quantile' :  0.1,   # quantile for the threshold calculation
        'thr_window'   :  1,     # window size for the threshold calculation
        'min_interval' :  1,     # minimum interval between spells
        'cdo_path'     :  '/usr/bin/cdo' # path to the cdo executable
    }

    @property
    def direction(self):
        # this is the direction of the index, -1 for deficits (LFI, Cold waves), 1 for surplusses (Heat waves)
        # must be implemented in the subclass
        raise NotImplementedError
    
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'parameters'): 
            self.parameters = ['threshold'] + self.ddi_names
        super().__init__(*args, **kwargs)

    def make_parameters(self,
                        history: TimeRange|Iterable[datetime],
                        timesteps_per_year: int) -> None:

        if isinstance(history, tuple) or isinstance(history, list):
            history = TimeRange(history[0], history[1])

        self.log.info(f'Calculating parameters for {history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}...')

        data_timesteps:list[FixedNTimeStep] = self.make_data_timesteps(history, timesteps_per_year)
        timesteps:list[FixedNTimeStep] = TimeRange('1900-01-01', '1900-12-31').days
        
        # make aggregated data for the parameters
        self.make_input_data(data_timesteps)

        # The only parameter we are calculating here is the Threshold
        self._parameters['threshold'].update(history_start = history.start, history_end = history.end, in_place = True)

        # check cases that have already been calculated
        cases_to_do = []
        for case in self.cases['opt']:
            this_ts_todo = self._parameters['threshold'].find_times(timesteps, rev = True, **case['tags'])
            if len(this_ts_todo) > 0:
                cases_to_do.append(case)

        # if nothing needs to be calculated, skip
        if len(cases_to_do) == 0: return
        self.log.info(f'  -Iterating through {len(cases_to_do)} cases with missing parameters.')

        for case in cases_to_do:
            variable = self._data.update(**case['tags'])
            thresholds = self.calc_thresholds_cdo(variable, history, case)
            for thr, thr_info in thresholds:
                this_timestep = Day.from_date(thr_info.pop('time'))
                metadata = case['options']
                metadata.update(thr_info)
                self._parameters['threshold'].write_data(thr, this_timestep, metadata = metadata, **case['tags'])

    def calc_thresholds_cdo(self,
                            variable: Dataset,
                            history: TimeRange,
                            case: dict) -> Generator[tuple[xr.DataArray, dict], None, None]:
        """
        This will only do the threshold calculation using CDO.
        """

        days = history.days

        threshold_info = {'reference_start': history.start.strftime('%Y-%m-%d'),
                          'reference_end':   history.end.strftime('%Y-%m-%d')}
        
        tmpdir = tempfile.mkdtemp()

        # save the data to a temporary file as netcdf with a time dimension
        days = history.days
        datasets = [variable.get_data(day, as_is=True, **case['tags']).squeeze().expand_dims('time').assign_coords(time=[day.start]) for day in days]
        combined = xr.concat(datasets, dim='time')
        data_nc = f'{tmpdir}/data.nc'
        combined.to_netcdf(data_nc)

        # calculate running max and min of data using CDO
        window_size = case['options']['thr_window']
        datamin_nc = f'{tmpdir}/datamin.nc'
        datamax_nc = f'{tmpdir}/datamax.nc'

        cdo_path = case['options']['cdo_path']
        cdo_cmd =  f'{cdo_path} ydrunmin,{window_size},rm=c {data_nc} {datamin_nc}'
        subprocess.run(cdo_cmd, shell = True)

        cdo_cmd = f'{cdo_path} ydrunmax,{window_size},rm=c {data_nc} {datamax_nc}'
        subprocess.run(cdo_cmd, shell = True)

        # calculate the thresholds
        threshold_quantile = int(case['options']['thr_quantile']*100)
        threshold_nc = f'{tmpdir}/threshold.nc'

        cdo_cmd = f'{cdo_path} ydrunpctl,{threshold_quantile},{window_size},rm=c,pm=r8 {data_nc} {datamin_nc} {datamax_nc} {threshold_nc}'
        subprocess.run(cdo_cmd, shell = True)

        # read the threshold data
        thresholds = xr.open_dataset(threshold_nc)
        
        # get the crs and write it to the threshold data
        thresholds_da = thresholds['__xarray_dataarray_variable__']
        crs = thresholds['spatial_ref'].attrs['crs_wkt']
        thresholds_da = thresholds['__xarray_dataarray_variable__'].rio.write_crs(crs, inplace = True)

        # add a 1-d "band" dimension
        thresholds_da = thresholds_da.expand_dims('band')
        
        # loop over the timesteps and yield the threshold
        days = thresholds_da.time.values
        for i, date in enumerate(days):
            time = datetime.fromtimestamp(date.astype('O') / 1e9)
            threshold_data = thresholds_da.isel(time = i).drop_vars('time')
            threshold_info.update({'time': time})
            yield threshold_data, threshold_info

        shutil.rmtree(tmpdir)

    def make_ddi(self,
                 time_range: Optional[TimeRange], 
                 timesteps_per_year: int,
                 thresholds: dict[int:Dataset],
                 cases: dict[int:list[dict]],
                 keep_historical = False) -> dict:

        """
        This function will calculate the DDI for the given time_range.
        DDI = deviation (current cumulative deficit/surplus),
              duration (of the current spell)
              interval (since the last spell)

        if keep_historical is True, all cumulative deficits/surpluses in the timeperiod will be kept and returned
        (this is useful for the LFI to calculate the lambda)

        This function assumes that we are starting from scratch, there is no existing DDI for any previous timestep.
        Use update_ddi to update the DDI from a previous timestep.
        """

        timesteps:list[FixedNTimeStep] = time_range.get_timesteps_from_tsnumber(timesteps_per_year)
        ts_ends = [ts.end for ts in timesteps]

        # get all the (daily) deviations for the history period
        input = self._data
        days = time_range.days
        all_deviations = self.get_deviations(input, thresholds, days)

        # set the starting conditions for each case we need to calculate
        start_ddi  = np.full((3,*input.get_template().shape), np.nan)
        ddi_dict = {k:{v['id']:start_ddi.copy() for v in v} for k,v in cases.items()}

        # set the cumulative deficit/surplus, we only need the average deficit to
        # calculate the lambda for the LFI, so we keep track of the cumulative drought
        # and the number of droughts
        cum_deviation_raw = np.zeros((2, *input.get_template().shape))
        cum_deviation_dict = {k:{v['id']:cum_deviation_raw.copy() for v in v} for k,v in cases.items()}

        i = 0
        for timestep, deviation_dict in all_deviations:
            for thrcase, deviation in deviation_dict.items():
                for case in cases[thrcase]:
                    cid = case['id']
                    # get the current conditions
                    ddi = ddi_dict[thrcase][cid]
                    # and the options
                    options = cases[thrcase][cid]['options']
                    # pool the deficit
                    ddi, cum_deviation = pool_deviation(deviation, options, current_conditions = ddi)
                    # update the ddi and cumulative drought
                    ddi_dict[thrcase][cid] = ddi
                    cum_deviation_dict[thrcase][cid] += cum_deviation
                    if timestep.end >= ts_ends[i]:
                        # save the current ddi
                        tags = cases[thrcase][cid]['tags']
                        tags.update(thresholds[thrcase].tags)
                        metadata = cases[thrcase][cid]['options']
                        self.save_ddi(ddi, timesteps[i], metadata = metadata, **tags)
                        i += 1

        if keep_historical:
            return cum_deviation_dict

    def update_ddi(self,
                   ddi_start: Optional[np.ndarray],
                   ddi_time: TimeStep,
                   time: TimeStep,
                   case: dict,
                   history: TimeRange) -> np.ndarray:
        
        """
        This function will update the DDI from the previous timestep.
        It is used to calculate the Threshold Based index rather than the parameters.
        """

        tags = case['tags']
        tags.update({'history_start': history.start, 'history_end': history.end})

        # get the threshold
        threshold = self._parameters['threshold'].update(**tags)

        # get the deviations from the last available DDI to the current time
        input = self._data
        start = (ddi_time + 1).start
        end = time.end
        missing_days = TimeRange(start, end).days
        all_deviations = self.get_deviations(input, {0:threshold}, missing_days)

        # get the options
        options = case['options']

        if ddi_start is None:
            ddi_start = np.full((3, *input.get_template().shape), np.nan)
        
        # pool the deficit
        ddi = ddi_start
        for time, deviation_dict in all_deviations:
            deviation = deviation_dict[0]    
            ddi, _ = pool_deviation(deviation, options, current_conditions = ddi)

        return ddi

    def get_ddi(self, time: TimeStep,  history: TimeRange, case: dict) -> xr.DataArray:

        tags = case['tags']
        tags.update({'history_start': history.start, 'history_end': history.end})

        ddi_vars = [self._parameters[par].update(**tags) for par in self.parameters if par in self.ddi_names]
        ddi_ts, ddi_start = get_recent_ddi(time, ddi_vars)
        # if there is no ddi, we need to calculate it from scratch,
        # otherwise we can just use the most recent one and update from there
        ddi = None
        if ddi_ts is None:
            ddi_date = history.start - timedelta(days = 1)
            ddi_ts = type(time).from_date(ddi_date)
        elif ddi_ts == time:
            ddi = ddi_start

        if ddi is None:
            # calculate the current deviation
            ddi = self.update_ddi(ddi_start, ddi_ts, time, case, history)

        # save the ddi for the next timestep
        self.save_ddi(ddi, time, metadata = case['options'], **tags)

        return ddi

    def save_ddi(self, ddi: np.ndarray, time: TimeStep, metadata:dict = {}, **kwargs) -> None:
        # now let's save the ddi at the end of the history period
        for i in range(len(self.ddi_names)):
            data = ddi[i]
            output = self._parameters[self.ddi_names[i]].update(**kwargs)
            output.write_data(data, time, metadata = metadata, **kwargs)

    def get_deviations(self, variable: Dataset,
                       threshold: dict[int:Dataset],
                       timesteps: list[TimeStep]
                       ) -> Generator[tuple[TimeStep, dict[int:np.ndarray]], None, None]:
        """
        This function will calculate the deviations for the given timesteps.
        thresholds allows to specify different thresholds for different cases.
        thresholds = {case1: threshold1}
        direction is -1 for deficits (LFI, Cold waves), 1 for surplusses (Heat waves)
        the output is a generator that yields the time and the deviations for each case
        the deviations are a dictionary with the same keys as the thresholds
        """

        for ts in timesteps:
            data = variable.get_data(ts)
            thresholds = {case: th.get_data(ts) for case, th in threshold.items()}
            deviation = {}
            for case, thr in thresholds.items():
                this_deviation = (data.values - thr.values) * self.direction
                this_deviation[this_deviation < 0] = 0
                deviation[case] = this_deviation#np.squeeze(this_deficit)

            yield ts, deviation

class LFI(DRYESThrBasedIndex):
    name = 'LFI (Low Flow Index)'

    default_options = {
        'thr_quantile' :  0.05, # quantile for the threshold calculation
        'thr_window'   : 31,    # window size for the threshold calculation
        'min_duration' :  5,    # minimum duration of a spell
        'min_interval' : 10,    # minimum interval between spells
        'min_nevents'  :  5,     # minimum number of events in the historic period to calculate lambda (LFI only)
        'cdo_path'     :  '/usr/bin/cdo'
    }

    direction = -1

    ddi_names = ['deficit', 'duration', 'interval']

    def __init__(self, *args, **kwargs):
        self.parameters = ['threshold', 'lambda'] + self.ddi_names
        super().__init__(*args, **kwargs)
    
    def make_parameters(self, history: TimeRange|Iterable[datetime], timesteps_per_year: int):

        if isinstance(history, tuple) or isinstance(history, list):
            history = TimeRange(history[0], history[1])

        # the thresholds are calculated from the superclass
        super().make_parameters(history, 365)

        # now we need to calculate the lambda
        # let's separate the cases based on the threshold parameters
        # the deficits and the lambdas
        opt_groups = [['thr_quantile', 'thr_window'],
                      ['min_duration', 'min_interval', 'min_nevents']]
        thr_cases, lam_cases = make_case_hierarchy(self.cases['opt'], opt_groups)

        # update the history in all parameters
        parlambda = self._parameters['lambda'].update(history_start = history.start,
                                                      history_end   = history.end)
        
        cases_to_calc_lambda = {}
        thresholds = {}
        n = 0
        for thrcase in thr_cases:
            this_thrid = thrcase['id']
            this_thresholds = self._parameters['threshold'].update(**thrcase['tags'])
            thresholds[this_thrid] = this_thresholds
            cases_to_calc_lambda[this_thrid] = []
            for lamcase in lam_cases[this_thrid]:
                this_lamid = lamcase['id']
                #this_lambda_path = substitute_values(raw_lambda_path, lamcase['tags'])
                if not parlambda.check_data(**lamcase['tags']):
                    cases_to_calc_lambda[this_thrid].append(lamcase)
                    n+=1

        all_cases = sum(len(v) for v in lam_cases)
        self.log.info(f'  - lambda: {all_cases-n}/{all_cases} with lambda alredy computed')
        if n == 0: return

        # Now do the DDI calculation, this will be useful for lambda later
        cum_drought_dict = self.make_ddi(history, timesteps_per_year, thresholds, cases_to_calc_lambda, keep_historical=True)

        for thrcase in thr_cases:
            this_thrid = thrcase['id']
            if this_thrid not in cases_to_calc_lambda.keys(): continue

            for lamcase in lam_cases[this_thrid]:
                this_lamid = lamcase['id']
                all_lamids = [v['id'] for v in lam_cases[this_thrid]]
                if this_lamid not in all_lamids: continue

                this_cum_drought = cum_drought_dict[this_thrid][this_lamid]
                min_nevents = lamcase['options']['min_nevents']

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_deficit = this_cum_drought[0] / this_cum_drought[1]
                    mean_deficit[this_cum_drought[1] < min_nevents] = np.nan
                    lambda_data = 1/mean_deficit
                
                metadata = lamcase['options']
                parlambda.write_data(lambda_data, metadata = metadata, **lamcase['tags'])

    def calc_index(self, time,  history: TimeRange, case: dict) -> xr.DataArray:
        # take the current ddi from the superclass
        ddi = self.get_ddi(time, history, case)

        deficit  = ddi[0]
        duration = ddi[1]
        min_duration = case['options']['min_duration']
        deficit[duration < min_duration] = 0

        # get the lambda
        lambda_data = self._parameters['lambda'].get_data(time, **case['tags'])

        lfi_data = 1 - np.exp(-lambda_data * deficit)
        lfi_info = lambda_data.attrs

        parents = {'lambda': lambda_data, 'duration': duration, 'interval': ddi[2]}
        lfi_info['parents'] = parents

        return lfi_data, lfi_info

class HCWI(DRYESThrBasedIndex):
    name = 'HCWI (Heat or Cold Wave Indices)'
    default_options = {
        'thr_window'   : 11,    # window size for the threshold calculation
        'min_interval' :  1,    # minimum interval between spells
        'min_duration' :  3,    # minimum duration of a spell
        'Ttype'        : {'max' : 'max', 'min' : 'min'},  # type of temperature use both max and min temperature
        # this entails that both the max and min temperature need to be above (below) the threshold to have a heat (cold) wave
        'cdo_path'     :  '/usr/bin/cdo'
    }

    ddi_names = ['intensity', 'duration', 'interval']

    def make_index(self, current: TimeRange, reference: TimeRange, timesteps_per_year: int):

        # remove the min and max (Ttype) temperature from the options and run the superclass method.

        options_without_minmax = self.default_options.copy()
        options_without_minmax.pop('Ttype')

        cases_without_minmax = options_to_cases(options_without_minmax)
        self.cases['opt'] = cases_without_minmax

        super().make_index(current, reference, timesteps_per_year)

    def calc_index(self, time,  history: TimeRange, case: dict) -> xr.DataArray:

        ddi = self.get_ddi(time, history, case)

        intensity  = ddi[0]
        duration = ddi[1]
        min_duration = case['options']['min_duration']

        intensity[duration < min_duration] = 0
        HWI_info = case['options']

        HWI_info['parents'] = {'duration': duration, 'interval': ddi[2]}

        return intensity, HWI_info
        
    def get_deviations(self,
                       variable:  Dataset,
                       threshold: dict[int:Dataset],
                       timesteps: list[TimeStep]) -> Generator[tuple[TimeStep, dict[int:np.ndarray]],
                                                               None, None]:
        """
        This function will calculate the deviations for the given timesteps.
        thresholds allows to specify different thresholds for different cases.
        thresholds = {case1: threshold1}
        direction is -1 for deficits (LFI, Cold waves), 1 for surplusses (Heat waves)
        the output is a generator that yields the time and the deviations for each case
        the deviations are a dictionary with the same keys as the thresholds
        """

        Ttypes = ['max', 'min']

        variables =  [variable.update(Ttype = Ttype) for Ttype in Ttypes]
        thresholds = [{case: th.update(Ttype = Ttype) for case, th in threshold.items()} for Ttype in Ttypes]

        deviations_Tmax = super().get_deviations(variables[0], thresholds[0], timesteps)
        deviations_Tmin = super().get_deviations(variables[1], thresholds[1], timesteps)

        for (ts_max, deviation_Tmax), (ts_min, deviation_Tmin) in zip(deviations_Tmax, deviations_Tmin):
            if ts_max != ts_min:
                raise ValueError(f'The time steps for Tmax {ts_max} and Tmin {ts_min} do not match')
            deviation = {}
            for case in deviation_Tmax.keys():
                deviation[case] = np.where((deviation_Tmax[case] == 0) | (deviation_Tmin[case] == 0), 0, (deviation_Tmax[case] + deviation_Tmin[case]) / 2)

            yield ts_max, deviation

class HWI(HCWI):
    name = 'HWI (Heat Wave Index)'
    direction = 1

    @property
    def default_options(self):
        defopts = super().default_options 
        defopts.update({'thr_quantile' : 0.9}) # quantile for the threshold calculation
        return defopts

class CWI(HCWI):
    name = 'CWI (Cold Wave Index)'
    direction = -1
    
    @property
    def default_options(self):
        defopts = super().default_options
        defopts.update({'thr_quantile' : 0.1}) # quantile for the threshold calculation
        return defopts

def pool_deviation(deviation: np.ndarray,
                   options: dict[str, float],
                   current_conditions: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    This function will pool the deviation for a single timestep.
    The inputs are:
    - the deviation for the current timestep (2-dim array)
    - the options for the pooling (min_duration, min_interval)
    - the current conditions (3-dim array, shape = (3, *deviation.shape))
        the first dimension is for the drought deviation, duration, and interval

    The outputs are:
    - the conditions after the computation (3-dim array, shape = (3, *deviation.shape))
        the first dimension is for the current cumulative deviation, duration, and interval
    - the cumulative deficit/surplus after the computation  (3-dim array, shape = (2, *deviation.shape))
        the first dimension is for the cumulative deficit/surplus and the number of events
    """

    # get the relevant options
    if 'min_duration' not in options.keys():
        options['min_duration'] = 1
    
    min_duration = options['min_duration']
    min_interval = options['min_interval']

    # initialize the output
    historical_cum_deviation = np.full((2, *deviation.shape), np.nan)

    # get the indices where we have data and where we have a deviation or a current condition
    any_data = ~np.isnan(deviation)
    #any_deviation = deviation>0
    #any_current = (current_conditions[0,:,:] > 0) | (current_conditions[1,:,:] > 0) | (current_conditions[2,:,:] > min_interval)
    #indices = np.where(np.all([any_data, np.any([any_deviation, any_current], axis=0)], axis = 0))
    indices = np.where(any_data)
    for b,x,y in zip(*indices):
        this_deviation = deviation[0,x,y]
        this_current_conditions = current_conditions[:,b,x,y]
        # pool the deficit for this pixel
        this_ddi, this_cum_deviation = pool_deviation_singlepixel(this_deviation,
                                                                  min_duration, min_interval,
                                                                  this_current_conditions)

        # update the output
        current_conditions[:,b,x,y] = this_ddi
        historical_cum_deviation[:,b,x,y] = this_cum_deviation

    return current_conditions, historical_cum_deviation

def pool_deviation_singlepixel(deviation: float,
                               min_duration: int = 1,
                               min_interval: int = 1,
                               current_conditions: np.ndarray = np.zeros(3)
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as pool_deviation, but for a single pixel.
    """
    # pool the deficit for a single pixel
    current_conditions[np.isnan(current_conditions)] = 0
    current_deviation, duration, interval = current_conditions

    cum_deviation  = 0
    ndroughts      = 0

    if deviation > 0:
        current_deviation += deviation    # add the deficit to the current deficit/surplus
        duration += 1         # increase the duration
        interval = 0          # reset the interval
    # if we don't have a deficit/surplus at this timestep
    else:
        # increase the interval from the last deficit/surplus
        # we do this first, becuase otherwise we overshoot the min_interval
        # also, this needs to happen regardless of whether we are in a drought/HC wave or not
        interval += 1
        # if we are in a drought or HC wave (accumulated deficit/surplus > 0)
        if current_deviation >= 0:
            # check if the drought/HC wave should end here
            # this is the case if it has been long enough from the last deficit/surplus
            if interval > min_interval:
                # end the drought
                this_cum_deviation  = current_deviation
                this_duration = duration
                # reset the counters
                current_deviation = 0
                duration = 0
                # check if the drought / HC wave should be saved
                # this is the case if the duration is long enough
                if this_duration >= min_duration:
                    # save the drought
                    ndroughts += 1
                    cum_deviation += this_cum_deviation
    
    cum_spell = np.array([cum_deviation, ndroughts])
    final_conditions = np.array([current_deviation, duration, interval])
    return final_conditions, cum_spell

def get_recent_ddi(time: TimeStep, ddi:Iterable[Dataset], **kwargs) -> tuple[Optional[TimeStep], Optional[np.ndarray]]:
    """
    returns the most recent ddi for the given time
    """

    if not all([var.get_start(**kwargs) for var in ddi]):
        return None, None

    while not all([var.find_times([time], **kwargs) for var in ddi]):
        time -= 1
        if (time + 1).end < ddi[0].get_start(**kwargs):
            return None, None

    ddi = np.stack([var.get_data(time,**kwargs) for var in ddi], axis = 0)

    return time, ddi
