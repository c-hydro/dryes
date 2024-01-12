from datetime import datetime
import numpy as np
import xarray as xr
import warnings

from typing import List

from .dryes_index import DRYESIndex

from ..io import IOHandler
from ..time_aggregation import aggregation_functions as agg

from ..utils.time import TimeRange

class DRYESAnomaly(DRYESIndex):
    index_name = 'anomaly'
    default_options = {
        'agg_fn' : {'Agg1': agg.average_of_window(1, 'months')},
        'type'   : 'empiricalzscore'
    }
    
    @property
    def parameters(self):
        opt_cases = self.cases['opt']
        all_types = [case['options']['type'] for case in opt_cases]
        return ['mean', 'std'] if 'empiricalzscore' in all_types else ['mean']

    def calc_parameters(self,
                        time: datetime,
                        variable: IOHandler,
                        history: TimeRange,
                        par_and_cases: dict[str:List[int]]) -> dict[str:dict[int:xr.DataArray]]:
        """
        Calculates the parameters for the Anomaly.
        par_and_cases is a dictionary with the following structure:
        {par: [case1, case2, ...]}
        indicaing which cases from self.cases['opt'] need to be calculated for each parameter.

        The output is a dictionary with the following structure:
        {par: {case1: parcase1, case2: parcase1, ...}}
        where parcase1 is the parameter par for case1 as a xarray.DataArray.
        """
        
        history_years = range(history.start.year, history.end.year + 1)
        all_dates  = [datetime(year, time.month, time.day) for year in history_years]
        data_dates = [date for date in all_dates if date >= history.start and date <= history.end]

        data_ = [variable.get_data(time) for time in data_dates]
        data = np.stack(data_, axis = 0)

        output = {}
        # calculate the parameters
        parameters = par_and_cases.keys()
        # we are using a warning catcher here because np.nanmean and np.nanstd will throw a warning if all values are NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # mean
            if 'mean' in parameters:
                mean_data = np.nanmean(data, axis = 0)
                output['mean'] = {0:mean_data} # -> we save this as case 0 because it is the same for all cases
            # std
            if 'std' in parameters:
                std_data = np.nanstd(data, axis = 0)
                output['std'] = {0:std_data} # -> we save this as case 0 because it is the same for all cases
        return output
    
    def calc_index(self, time,  history: TimeRange, case: dict) -> xr.DataArray:
        # load the data for the index
        data = self._data.get_data(time, **case['tags'])

        # load the parameters
        if 'history_start' not in case['tags']:
            case['tags']['history_start'] = history.start
        if 'history_end' not in case['tags']:
            case['tags']['history_end'] = history.end
        parameters = {par: p.get_data(time, **case['tags']) for par, p in self._parameters.items()}

        # calculate the index
        mean = parameters['mean']
        if case['options']['type'] == 'empiricalzscore':
            std  = parameters['std']
            anomaly_data = (data - mean) / std
        elif case['options']['type'] == 'percentdelta':
            anomaly_data = (data - mean) / mean * 100
        elif case['options']['type'] == 'absolutedelta':
            anomaly_data = data - mean
        else:
            raise ValueError(f"Unknown type {case['options']['type']} for anomaly index.")
        
        return anomaly_data