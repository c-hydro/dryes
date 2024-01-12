from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import warnings
import glob
import os

from typing import List, Optional

from .dryes_index import DRYESIndex

from ..io import IOHandler

from ..utils.time import TimeRange, create_timesteps
from ..utils.parse import substitute_values, make_case_hierarchy
from ..utils.log import log

class DRYESLFI(DRYESIndex):
    index_name = 'LFI'
    # this flag is used to determine if we need to process all timesteps continuously
    # or if we can have gaps in the processing
    # the LFI always works continuously (for other indices, it depends on the time aggregation)
    iscontinuous = True

    # default options for the LFI (this is based on the EDO LFI factsheet)
    default_options = {
        'thr_quantile' :  0.05,
        'thr_window'   : 31,
        'min_duration' :  5,
        'min_interval' : 10,
        'min_nevents'  :  5
    }

    # parameters for the LFI
    parameters = ('Qthreshold', 'lambda', 'Ddeficit', 'duration', 'interval')

    def make_parameters(self, history: TimeRange, timesteps_per_year: int):
        # for the LFI, we need to do this separately from the superclass, because parameters work a little differently
        # the threshold is calculated from the superclass
        parameters_all = self.parameters
        self.parameters = ('Qthreshold',)
        super().make_parameters(history, 365)
        self.parameters = parameters_all

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
            this_thresholds = self._parameters['Qthreshold'].update(**thrcase['tags'])
            thresholds[this_thrid] = this_thresholds
            cases_to_calc_lambda[this_thrid] = []
            for lamcase in lam_cases[this_thrid]:
                this_lamid = lamcase['id']
                #this_lambda_path = substitute_values(raw_lambda_path, lamcase['tags'])
                if not parlambda.check_data(**lamcase['tags']):
                    cases_to_calc_lambda[this_thrid].append(lamcase)
                    n+=1

        all_cases = sum(len(v) for v in lam_cases)
        log(f'  - lambda: {all_cases-n}/{all_cases} with lambda alredy computed')
        if n == 0: return

        # Now do the DDI calculation, this will be useful for lambda later
        cum_drought_dict = self.make_ddi(history, timesteps_per_year, thresholds, cases_to_calc_lambda, keep_cumdeficit=True)

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
                
                parlambda.write_data(lambda_data, **lamcase['tags'])

    def make_ddi(self,
                 time_range: Optional[TimeRange], 
                 timesteps_per_year: int,
                 thresholds: dict[int:IOHandler],
                 cases: dict[int:list[dict]],
                 keep_cumdeficit = False) -> dict:
    
        timesteps = create_timesteps(time_range.start, time_range.end, timesteps_per_year)

        # get all the deficits for the history period
        #input_path = self.output_paths['data']
        input = self._data
        #test_period = TimeRange(history.start, history.start + timedelta(days = 365))
        all_deficits = get_deficits(input, thresholds, time_range)

        # set the starting conditions for each case we need to calculate lambda for
        start_ddi  = np.full((3,*input.template.shape), np.nan)
        ddi_dict = {k:{v['id']:start_ddi.copy() for v in v} for k,v in cases.items()}
        # ddi stands for drought_deficit, duration, interval

        # set the cumulative drought, we only need the average drougth deficit to
        # calculate the lambda, so we keep track of the cumulative drought
        # and the number of droughts
        cum_drought_raw = np.zeros((2, *input.template.shape))
        cum_drought_dict = {k:{v['id']:cum_drought_raw.copy() for v in v} for k,v in cases.items()}

        for time, deficit_dict in all_deficits:
            for thrcase, deficit in deficit_dict.items():
                for case in cases[thrcase]:
                    cid = case['id']
                    # get the current conditions
                    ddi = ddi_dict[thrcase][cid]
                    # and the options
                    options = cases[thrcase][cid]['options']
                    # pool the deficit
                    ddi, cum_drought = pool_deficit(deficit, options, current_conditions = ddi)
                    # update the ddi and cumulative drought
                    ddi_dict[thrcase][cid] = ddi
                    cum_drought_dict[thrcase][cid] += cum_drought
                    if time in timesteps:
                    # save the current ddi
                        tags = cases[thrcase][cid]['tags']
                        tags.update(thresholds[thrcase].tags)
                        self.save_ddi(ddi, time, tags)

        if keep_cumdeficit:
            return cum_drought_dict

    def update_ddi(self,
                   ddi_start: np.ndarray,
                   ddi_time: datetime,
                   time: datetime,
                   case: dict,
                   history: TimeRange) -> np.ndarray:
        
        tags = case['tags']
        tags.update({'history_start': history.start, 'history_end': history.end})

        # get the threshold
        threshold = self._parameters['Qthreshold'].update(**tags)

        # get the deficits
        input = self._data
        all_deficits = get_deficits(input, {0:threshold}, TimeRange(ddi_time, time))

        # get the options
        options = case['options']

        if ddi_start is None:
            ddi_start = np.full((3, *input.template.shape), np.nan)
        
        # pool the deficit
        ddi = ddi_start
        for time, deficit_dict in all_deficits:
            deficit = deficit_dict[0]    
            ddi, _ = pool_deficit(deficit, options, current_conditions = ddi)

        return ddi

    def calc_parameters(self,
                        time: datetime,
                        variable: IOHandler,
                        history: TimeRange,
                        par_and_cases: dict[str:List[int]]) -> dict[str:dict[int:xr.DataArray]]:
        """
        Calculates the parameters for the LFI. This will only do the threshold calculation.
        par_and_cases is a dictionary with the following structure:
        {par: [case1, case2, ...]}
        indicaing which cases from self.cases['opt'] need to be calculated for each parameter.

        The output is a dictionary with the following structure:
        {par: {case1: parcase1, case2: parcase1, ...}}
        where parcase1 is the parameter par for case1 as a xarray.DataArray.
        """

        history_years = range(history.start.year, history.end.year + 1)
        dates  = [datetime(year, time.month, time.day) for year in history_years]

        #input_path = in_path
        output = {'Qthreshold': {}} # this is the output dictionary

        # loop over all cases, let's do this in a smart way so that we don't have to load the data multiple times
        window_cases = {}
        for case in self.cases['opt']:
            if case['id'] not in par_and_cases['Qthreshold']: continue
            # get the window
            window = case['options']['thr_window']
            # add the case to the dictionary
            window_cases[window] = window_cases.get(window, []) + [case['id']]
        
        for window, cases in window_cases.items():
            halfwindow = np.floor(window/2)
            # get all dates within the window of this month and day in the reference period
            data_dates = []
            for date in dates:
                this_date = date - timedelta(days = halfwindow)
                while this_date <= date + timedelta(days = halfwindow):
                    if this_date >= history.start and this_date <= history.end:
                        data_dates.append(this_date)
                    this_date += timedelta(days = 1)
            
            # if there are no dates, skip
            if len(data_dates) == 0: continue
            
            # get the data for these dates
            data_ = [variable.get_data(time) for time in data_dates]
            data = np.stack(data_, axis = 0)

            # calculate the thresholds for each case that has this window size
            for case in cases:
                threshold_quantile = self.cases['opt'][case]['options']['thr_quantile']
                threshold_data = np.quantile(data, threshold_quantile, axis = 0)
                # save the threshold
                #threshold = data_template.copy(data = threshold_data)
                output['Qthreshold'][case] = threshold_data

        return output

    def calc_index(self, time,  history: TimeRange, case: dict) -> xr.DataArray:
        
        tags = case['tags']
        tags.update({'history_start': history.start, 'history_end': history.end})

        ddi = [self._parameters[par].update(**tags) for par in self.parameters if par in ['Ddeficit', 'duration', 'interval']]
        ddi_time, ddi_start = get_recent_ddi(time, ddi)
        # if there is no ddi, we need to calculate it from scratch,
        # otherwise we can just use the most recent one and update from there
        ddi = None
        if ddi_time == time:
            ddi = ddi_start
        elif ddi_time is None:
            ddi_time = history.start

        if ddi is None:
            # calculate the deficit
            ddi = self.update_ddi(ddi_start, ddi_time, time - timedelta(1), case, history)

        # save the ddi for the next timestep
        self.save_ddi(ddi, time, tags)
        
        deficit  = ddi[0]
        duration = ddi[1]
        min_duration = case['options']['min_duration']
        deficit[duration < min_duration] = 0

        # get the lambda
        lambda_data = self._parameters['lambda'].update(**tags).get_data(time)

        lfi_data = 1 - np.exp(-lambda_data * deficit)
        return lfi_data

    def save_ddi(self, ddi: np.ndarray, time: datetime, tags):
        # now let's save the ddi at the end of the history period
        ddi_names = ['Ddeficit', 'duration', 'interval']
        for i in range(3):
            data = ddi[i]
            output = self._parameters[ddi_names[i]].update(**tags)
            output.write_data(data, time)

def get_deficits(streamflow: IOHandler, threshold: dict[int:IOHandler], time_range:TimeRange) -> dict[int:np.ndarray]:
    for time in time_range:
        if (time.month, time.day) == (2,29): continue
        data = streamflow.get_data(time)
        thresholds = {case: th.get_data(time) for case, th in threshold.items()}
        deficit = {}
        for case, thr in thresholds.items():
            this_deficit = thr.values - data.values
            this_deficit[this_deficit < 0] = 0
            deficit[case] = this_deficit#np.squeeze(this_deficit)

        yield time, deficit

def pool_deficit(deficit: np.ndarray,
                 options: dict[str, float],
                 current_conditions: np.ndarray,
                    ) -> [np.ndarray, np.ndarray]:
    """
    This function will pool the deficit for a single timestep.
    The inputs are:
    - the deficit for the current timestep (2-dim array)
    - the options for the LFI (min_duration, min_interval)
    - the current conditions (3-dim array, shape = (3, *deficit.shape))
        the first dimension is for the drought deficit, duration, and interval

    The outputs are:
    - the conditions after the computation (3-dim array, shape = (3, *deficit.shape))
        the first dimension is for the drought deficit, duration, and interval
    - the cumulative drought deficit after the computation  (3-dim array, shape = (2, *deficit.shape))
        the first dimension is for the cumulative drought deficit and the number of droughts
    """

    # get the relevant options
    min_duration = options['min_duration']
    min_interval = options['min_interval']

    # initialize the output
    cum_drought = np.full((2, *deficit.shape), np.nan)

    # get the indices where we have a streamflow deficit
    indices = np.where(~np.isnan(deficit))
    for b,x,y in zip(*indices):
        this_deficit = deficit[0,x,y]
        this_current_conditions = current_conditions[:,b,x,y]
        # pool the deficit for this pixel
        this_ddi, this_cum_drought = pool_deficit_singlepixel(this_deficit,
                                                              min_duration, min_interval,
                                                              this_current_conditions)

        # update the output
        current_conditions[:,b,x,y] = this_ddi
        cum_drought[:,b,x,y] = this_cum_drought

    return current_conditions, cum_drought

def pool_deficit_singlepixel(deficit: float,
                             min_duration: int = 5,
                             min_interval: int = 10,
                             current_conditions: np.ndarray = np.zeros(3)
                             ) -> [np.ndarray, np.ndarray]:
    """
    Same as pool_deficit, but for a single pixel.
    """
    # pool the deficit for a single pixel
    current_conditions[np.isnan(current_conditions)] = 0
    drought_deficit, duration, interval = current_conditions

    cum_deficit  = 0
    ndroughts    = 0

    if deficit > 0:
        drought_deficit += deficit    # add the deficit to the current drought
        duration += 1         # increase the duration
        interval = 0          # reset the interval
    # if we don't have a streamflow deficit at this timestep
    else:
        # increase the interval from the last deficit
        # we do this first, becuase otherwise we overshoot the min_interval
        # also, this needs to happen regardless of whether we are in a drought or not
        interval += 1
        # if we are in a drought (accumulated deficit > 0)
        if drought_deficit > 0:
            # check if the drought should end here
            # this is the case if it has been long enough from the last deficit
            if interval >= min_interval:
                # end the drought
                this_drought  = drought_deficit
                this_duration = duration
                # reset the counters
                drought_deficit = 0
                duration = 0
                # check if the drought should be saved
                # this is the case if the duration is long enough
                if this_duration >= min_duration:
                    # save the drought
                    ndroughts += 1
                    cum_deficit += this_drought
    
    cum_drought = np.array([cum_deficit, ndroughts])
    final_conditions = np.array([drought_deficit, duration, interval])
    return final_conditions, cum_drought

def get_recent_ddi(time, ddi):
    """
    returns the most recent ddi for the given time
    """
    # deficit, duration, interval = ddi
    # ddi_vars  = ['Ddeficit', 'duration', 'interval']
    ddi_times = {}
    # ddi_paths  = {var: ddi_paths[i] for i, var in enumerate(ddi_vars)}
    for var in ddi:
        this_ddi_path = var.path()
        this_ddi_path_glob = this_ddi_path.replace('%Y', '*').\
                                           replace('%m', '*').\
                                           replace('%d', '*')
        all_ddi_paths = glob.glob(this_ddi_path_glob)
        times = []
        for path in all_ddi_paths:
            this_time = datetime.strptime(os.path.basename(path), os.path.basename(this_ddi_path))
            times.append(this_time)
        
        #times = [datetime.strptime(os.path.basename(path), this_ddi_path) for path in all_ddi_paths]
        ddi_times[var.name] = set(times)
    
    # get the timesteps common to all ddi variables
    common_times = set.intersection(*ddi_times.values())
    common_times_before_time = {t for t in common_times if t <= time}
    if common_times_before_time == set(): return None, None

    # get the most recent timestep
    recent_time = max(common_times_before_time)
    Ddeficit, duration, interval = [var.get_data(recent_time) for var in ddi]

    ddi = np.stack([Ddeficit, duration, interval], axis = 0)

    return recent_time, ddi




    # for t in range(deficit.shape[0]):
    #     # if we have a streamflow deficit at this timestep
    #     if deficit[t] > 0:
    #         drought += deficit[t] # add the deficit to the drought
    #         duration += 1         # increase the duration
    #         interval = 0          # reset the interval
    #     # if we don't have a streamflow deficit at this timestep
    #     else:
    #         # increase the interval from the last deficit
    #         # we do this first, becuase otherwise we overshoot the min_interval
    #         # also, this needs to happen regardless of whether we are in a drought or not
    #         interval += 1
    #         # if we are in a drought (accumulated deficit > 0)
    #         if drought > 0:
    #             # check if the drought should end here
    #             # this is the case if it has been long enough from the last deficit
    #             if interval >= min_interval:
    #                 # end the drought
    #                 this_drought  = drought
    #                 this_duration = duration
    #                 # reset the counters
    #                 drought = 0
    #                 duration = 0
    #                 # check if the drought should be saved
    #                 # this is the case if the duration is long enough
    #                 if this_duration >= min_duration:
    #                     # save the drought
    #                     droughts.append(this_drought)

    # return droughts, (drought, duration, interval)