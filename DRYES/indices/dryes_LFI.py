from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import warnings
import glob

from typing import List, Optional

from .dryes_index import DRYESIndex

from ..lib.time import TimeRange, doy_to_md
from ..lib.parse import substitute_values, make_case_hierarchy
from ..lib.io import get_data, check_data, save_dataarray_to_geotiff
from ..lib.log import log

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

    def make_parameters(self, history: TimeRange):
        # for the LFI, we need to do this separately from the superclass, because parameters work a little differently
        # the threshold is calculated from the superclass
        self.parameters = ('Qthreshold',)
        super().make_parameters(history)

        # let's separate the cases based on the threshold parameters
        # the deficits and the lambdas
        opt_groups = [['thr_quantile', 'thr_window'],
                      ['min_duration', 'min_interval', 'min_nevents']]
        thr_cases, lam_cases = make_case_hierarchy(self.cases['opt'], opt_groups)

        # first thing, figure out what cases we need to calculate lambda for
        raw_lambda_path = substitute_values(self.output_paths['lambda'],
                                            {'history_start' : history.start,
                                             'history_end'   : history.end})
        
        cases_to_calc_lambda = {}
        n = 0
        for thrcase in thr_cases:
            this_thrid = thrcase['id']
            cases_to_calc_lambda[this_thrid] = []
            for lamcase in lam_cases[this_thrid]:
                this_lamid = lamcase['id']
                this_lambda_path = substitute_values(raw_lambda_path, lamcase['tags'])
                if not check_data(this_lambda_path):
                    cases_to_calc_lambda[this_thrid].append(this_lamid)
                    n+=1
        
        all_cases = sum(len(v) for v in cases_to_calc_lambda.values())
        log(f'  - lambda: {all_cases-n}/{all_cases} with lambda alredy computed')
        if n == 0: return

        deficit_cases_hierarchy = [thr_cases, lam_cases]
        ddi_dict, cum_drought_dict = self.calc_deficit(history, deficit_cases_hierarchy, cases_to_calc_lambda)

        template = self.output_template
        for thrcase in thr_cases:
            this_thrid = thrcase['id']
            if this_thrid not in cases_to_calc_lambda.keys(): continue
            for lamcase in lam_cases[this_thrid]:
                this_lamid = lamcase['id']
                if this_lamid not in cases_to_calc_lambda[this_thrid]: continue
                this_lambda_path = substitute_values(raw_lambda_path, lamcase['tags'])
                this_cum_drought = cum_drought_dict[this_thrid][this_lamid]
                min_nevents = lamcase['options']['min_nevents']

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_deficit = this_cum_drought[0] / this_cum_drought[1]
                    mean_deficit[this_cum_drought[1] < min_nevents] = np.nan
                    lambda_data = 1/mean_deficit
                
                lambda_da = template.copy(data = np.expand_dims(lambda_data,0))
                metadata = {'reference_period': f'{history.start:%d/%m/%Y}-{history.end:%d/%m/%Y}',
                            'name': 'lambda',
                            'type': 'DRYES parameter',
                            'index': self.index_name}
                save_dataarray_to_geotiff(lambda_da, this_lambda_path, metadata)

                ddi_path_raw = self.output_paths['ddi']
                ddi_path = substitute_values(ddi_path_raw, lamcase['tags'])
                ddi = ddi_dict[this_thrid][this_lamid]
                ddi_time = history.end + timedelta(days = 1)
                self.save_ddi(ddi, ddi_time, ddi_path)

    def calc_deficit(self, history: TimeRange,
                     deficit_cases_hierarchy: List[List[dict[str, dict]]],
                     cases_to_calc_deficit: dict[int:List[int]],
                     ddi_time:  Optional[TimeRange] = None,
                     start_ddi: Optional[np.ndarray] = None,
                     quiet = False) -> [dict, dict]:

        thr_cases, def_cases = deficit_cases_hierarchy

        # then we need to figure out the paths for the thresholds
        theshold_path_raw = substitute_values(self.output_paths['parameters'],
                                                {'par': 'Qthreshold',
                                                'history_start' : history.start,
                                                'history_end'   : history.end})        
        threshold_paths = {i: substitute_values(theshold_path_raw, case['tags'])
                           for i, case in enumerate(thr_cases) if i in cases_to_calc_deficit.keys()}

        # get all the deficits for the history period
        input_path = self.output_paths['data']
        #test_period = TimeRange(history.start, history.start + timedelta(days = 365))
        if ddi_time is None: ddi_time = history
        all_deficits = get_deficits(input_path, threshold_paths, ddi_time)

        # set the starting conditions for each case we need to calculate lambda for
        if start_ddi is None: start_ddi  = np.zeros((3,*self.input_variable.grid.shape))
        ddi_dict = {k:{vv:start_ddi.copy() for vv in v} 
                                           for k,v in cases_to_calc_deficit.items()}
        # ddi stands for drought_deficit, duration, interval

        # set the cumulative drought, we only need the average droguth deficit to
        # calculate the lambda, so we keep track of the cumulative drought
        # and the number of droughts
        cum_drought_raw = np.zeros((2, *self.input_variable.grid.shape))
        cum_drought_dict = {k:{vv:cum_drought_raw.copy() for vv in v}
                                             for k,v in cases_to_calc_deficit.items()}

        if not quiet:
            log(f'  - Computing streamflow deficits:')
        # loop over all timesteps
        for time, deficit_dict in all_deficits:
            if time.day == 1 and not quiet: log(f'   - {time:%Y-%m-%d}')
            for thrcase, deficit in deficit_dict.items():
                for case in cases_to_calc_deficit[thrcase]:
                    # get the current conditions
                    ddi = ddi_dict[thrcase][case]
                    # and the options
                    options = def_cases[thrcase][case]['options']
                    # pool the deficit
                    ddi, cum_drought = pool_deficit(deficit, options, current_conditions = ddi)
                    # update the ddi and cumulative drought
                    ddi_dict[thrcase][case] = ddi
                    cum_drought_dict[thrcase][case] += cum_drought

        return ddi_dict, cum_drought_dict

    def calc_parameters(self, dates: List[datetime],
                        par_and_cases: dict[str:List[int]],
                        reference: TimeRange) -> dict[str:dict[int:xr.DataArray]]:
        """
        Calculates the parameters for the LFI. This will only do the threshold calculation.
        par_and_cases is a dictionary with the following structure:
        {par: [case1, case2, ...]}
        indicaing which cases from self.cases['opt'] need to be calculated for each parameter.

        The output is a dictionary with the following structure:
        {par: {case1: parcase1, case2: parcase1, ...}}
        where parcase1 is the parameter par for case1 as a xarray.DataArray.
        """

        input_path = self.output_paths['data']
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
            all_dates = []
            for date in dates:
                this_date = date - timedelta(days = halfwindow)
                while this_date <= date + timedelta(days = halfwindow):
                    if this_date >= reference.start and this_date <= reference.end:
                        all_dates.append(this_date)
                    this_date += timedelta(days = 1)
            
            # if there are no dates, skip
            if len(all_dates) == 0: continue
            # get the data for these dates
            data = [get_data(input_path, date) for date in all_dates]
            data_template = self.output_template
            data = np.stack(data, axis = 0)

            # calculate the thresholds for each case that has this window size
            for case in cases:
                threshold_quantile = self.cases['opt'][case]['options']['thr_quantile']
                threshold_data = np.quantile(data, threshold_quantile, axis = 0)
                # save the threshold
                threshold = data_template.copy(data = threshold_data)
                output['Qthreshold'][case] = threshold

        return output

    def calc_lambda(self, history: TimeRange) -> xr.DataArray:
        pass

    def calc_index(self, time,  history: TimeRange, case: dict) -> xr.DataArray:
        # calculate the current deficit
        ddi_path = substitute_values(self.output_paths['ddi'], case['tags'])
        # get the most recent ddi
        ddi_time, ddi_start = get_recent_ddi(time, ddi_path)
        # if there is no ddi, we need to calculate it from scratch,
        # otherwise we can just use the most recent one and update from there
        ddi = None
        if ddi_time == time:
            ddi = ddi_start
        elif ddi_time is None:
            ddi_time_range = TimeRange(history.start, time - timedelta(days = 1))
        else:
            ddi_time_range = TimeRange(ddi_time, time - timedelta(days = 1))

        if ddi is None:
            # calculate the deficit
            deficit_cases_hierarchy = [[case], [[case]]]
            cases_to_calc = {0: [0]}
            ddi_dict, _ = self.calc_deficit(history, deficit_cases_hierarchy, cases_to_calc, ddi_time_range, ddi_start, quiet=True)
            ddi = ddi_dict[0][0]
        
        deficit  = ddi[0]
        duration = ddi[1]
        min_duration = case['options']['min_duration']
        deficit[duration < min_duration] = 0

        # save the ddi for the next timestep
        self.save_ddi(ddi, time, ddi_path)

        # get the lambda
        lambda_path_raw = substitute_values(self.output_paths['lambda'], {'history_start': history.start, 'history_end': history.end})
        lambda_path = substitute_values(lambda_path_raw, case['tags'])
        lambda_data = get_data(lambda_path).values

        lfi_data = 1 - np.exp(-lambda_data * deficit)
        output_template = self.output_template
        lfi = output_template.copy(data = lfi_data)
        return lfi

    def save_ddi(self, ddi: np.ndarray, time: datetime, path: str):
        # now let's save the ddi at the end of the history period
        ddi_names = ['deficit', 'duration', 'interval']
        path = time.strftime(path)
        for i in range(3):
            data = ddi[i]
            mask = self.input_variable.grid.mask
            data[~mask] = np.nan
            ddi_da = self.output_template.copy(data = np.expand_dims(data, 0))
            metadata = {
                    'name': ddi_names[i],
                    'type': 'DRYES parameter',
                    'index': self.index_name,
                    'time': time.strftime('%Y-%m-%d')}
            output_file = path.format(ddi_var = ddi_names[i])
            save_dataarray_to_geotiff(ddi_da, output_file, metadata)

def get_deficits(streamflow: str, threshold: dict[int:str], time_range:TimeRange) -> dict[int:np.ndarray]:
    for time in time_range:
        if (time.month, time.day) == (2,29): continue
        data = get_data(streamflow, time).values
        thresholds = {case: get_data(thr, time).values for case, thr in threshold.items()}
        deficit = {}
        for case, thr in thresholds.items():
            this_deficit = thr - data
            this_deficit[this_deficit < 0] = 0
            deficit[case] = np.squeeze(this_deficit)

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
    cum_drought = np.zeros((2, *deficit.shape))

    # get the indices where we have a streamflow deficit
    indices = np.where(~np.isnan(deficit))
    for x,y in zip(*indices):
        this_deficit = deficit[x,y]
        this_current_conditions = current_conditions[:,x,y]
        # pool the deficit for this pixel
        this_ddi, this_cum_drought = pool_deficit_singlepixel(this_deficit,
                                                              min_duration, min_interval,
                                                              this_current_conditions)

        # update the output
        current_conditions[:,x,y] = this_ddi
        cum_drought[:,x,y] = this_cum_drought

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

def get_recent_ddi(time, ddi_path):
    """
    returns the most recent ddi for the given time
    """
    ddi_vars  = ['deficit', 'duration', 'interval']
    ddi_times = {var: set() for var in ddi_vars}
    ddi_path  = {var: substitute_values(ddi_path, {'ddi_var': var}) for var in ddi_vars}
    for var in ddi_vars:
        this_ddi_path = ddi_path[var]
        this_ddi_path_glob = this_ddi_path.replace('%Y', '*').\
                                         replace('%m', '*').\
                                         replace('%d', '*')
        all_ddi_paths = glob.glob(this_ddi_path_glob)
        times = [datetime.strptime(path, this_ddi_path) for path in all_ddi_paths]
        ddi_times[var] = set(times)
    
    # get the timesteps common to all ddi variables
    common_times = set.intersection(*ddi_times.values())
    common_times_before_time = {t for t in common_times if t <= time}
    if common_times_before_time == set(): return None, None

    # get the most recent timestep
    recent_time = max(common_times_before_time)
    deficit, duration, interval = [get_data(path, recent_time).values for path in ddi_path.values()]

    ddi = np.squeeze(np.stack([deficit, duration, interval], axis = 0))

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