from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import warnings

from typing import List

from .dryes_index import DRYESIndex

from ..time_aggregation import aggregation_functions as agg

from ..tools.timestepping import TimeRange
from ..tools.timestepping.timestep import TimeStep
from ..tools.data import Dataset

class DRYESAnomaly(DRYESIndex):
    index_name = 'anomaly'
    default_options = {
        'agg_fn'       : {'Agg1': agg.average_of_window(1, 'months')},
        'type'         : 'empiricalzscore',
        'min_reference': 1,
        #'min_std'      : 0.01 #TODO: implement this as an option, for now it is hardcoded in the code
    }
    
    @property
    def parameters(self):
        opt_cases = self.cases['opt']
        all_types = [case['options']['type'] for case in opt_cases]
        return ['mean', 'std'] if 'empiricalzscore' in all_types else ['mean']

    def calc_parameters(self,
                        time: TimeStep,
                        variable: Dataset,
                        history: TimeRange,
                        par_and_cases: dict[str:List[int]]) -> tuple[dict[str:dict[int:np.ndarray]], dict]:
        """
        Calculates the parameters for the Anomaly.
        par_and_cases is a dictionary with the following structure:
        {par: [case1, case2, ...]}
        indicaing which cases from self.cases['opt'] need to be calculated for each parameter.

        2 outputs are returned:
        - a dictionary with the following structure:
            {par: {case1: parcase1, case2: parcase1, ...}}
            where parcase1 is the parameter par for case1 as a numpy.ndarray.
        - a dictionary with info about parameter calculations.
        """
        
        data_timesteps = time.get_history_timesteps(history)
        data_ = [variable.get_data(time) for time in data_timesteps if variable.check_data(time)]
        data = np.stack(data_, axis = 0)

        pardata = {}
        # calculate the parameters
        parameters = par_and_cases.keys()
        # we are using a warning catcher here because np.nanmean and np.nanstd will throw a warning if all values are NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # mean
            if 'mean' in parameters:
                mean_data = np.nanmean(data, axis = 0)
                pardata['mean'] = mean_data
                #output['mean'] = {0:mean_data} # -> we save this as case 0 because it is the same for all cases
            # std
            if 'std' in parameters:
                std_data = np.nanstd(data, axis = 0)
                pardata['std'] = std_data
                #output['std'] = {0:std_data} # -> we save this as case 0 because it is the same for all cases
        
        output = {}
        # count how many years of data we have for each pixel
        valid_data = np.sum(np.isfinite(data), axis = 0)
        for par in parameters:
            this_par_cases = par_and_cases[par]
            output[par] = {}
            for case in this_par_cases:
                this_min_reference = self.cases['opt'][case]['options']['min_reference']
                this_par_data = np.where(valid_data >= this_min_reference, pardata[par], np.nan)
                output[par][case] = this_par_data

        # get the metadata
        data_dates = [variable.get_time_signature(time) for time in data_timesteps]
        data_dates = ', '.join([date.strftime('%Y-%m-%d') for date in data_dates])
        par_info = {'reference_dates': data_dates,
                    'reference_start': history.start.strftime('%Y-%m-%d'),
                    'reference_end':   history.end.strftime('%Y-%m-%d')}

        if hasattr(data_[0], 'attrs'):
            data_info = data_[0].attrs
            if 'agg_type' in data_info:
                par_info['agg_type'] = data_info['agg_type']

        return output, par_info
    
    def calc_index(self, time: TimeStep,  history: TimeRange, case: dict) -> tuple[np.ndarray, dict]:
        """
        Calculates the index for the given time and reference period (history).
        Returns the index as a numpy.ndarray and a dictionary of metadata, if any.
        """

        # load the data for the index - allow for missing data
        if self._data.check_data(time, **case['tags']):
            data = self._data.get_data(time, **case['tags'])
        else:
            return self._data.template, {"NOTE": "missing data"}

        # load the parameters
        if 'history_start' not in case['tags']:
            case['tags']['history_start'] = history.start
        if 'history_end' not in case['tags']:
            case['tags']['history_end'] = history.end
        # parameters for 29th February are actually those for 28th February
        #par_time = time if time.month != 2 or time.day != 29 else time - timedelta(days = 1)
        parameters = {par: p.get_data(time, **case['tags']) for par, p in self._parameters.items()}

        # calculate the index
        mean = parameters['mean']
        if case['options']['type'] == 'empiricalzscore':
            std  = parameters['std']
            anomaly_data = (data - mean) / std
            #TODO: implement this as an option, for now it is hardcoded in the code
            anomaly_data = np.where(std < 0.01, np.nan, anomaly_data)
        elif case['options']['type'] == 'percentdelta':
            anomaly_data = (data - mean) / mean * 100
        elif case['options']['type'] == 'absolutedelta':
            anomaly_data = data - mean
        else:
            raise ValueError(f"Unknown type {case['options']['type']} for anomaly index.")
        
        if hasattr(data, 'attrs'):
            index_info = data.attrs
        else:
            index_info = {}

        if hasattr(list(parameters.values())[0], 'attrs'):
            index_info.update(list(parameters.values())[0].attrs)

        return anomaly_data, index_info