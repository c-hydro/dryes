from datetime import datetime
import numpy as np
import xarray as xr
import warnings

from typing import List, Optional

from .dryes_index import DRYESIndex

from ..time_aggregation import aggregation_functions as agg
from ..lib.time import TimeRange, ntimesteps_to_md
from ..lib.parse import substitute_values
from ..lib.log import log
from ..lib.io import get_data, check_data, save_dataarray_to_geotiff


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

    def calc_parameters(self, dates: List[datetime],
                        data_path: str,
                        par_and_cases: dict[str:List[int]],
                        reference: Optional[TimeRange]=None) -> dict[str:dict[int:xr.DataArray]]:
        """
        Calculates the parameters for the Anomaly.
        par_and_cases is a dictionary with the following structure:
        {par: [case1, case2, ...]}
        indicaing which cases from self.cases['opt'] need to be calculated for each parameter.

        The output is a dictionary with the following structure:
        {par: {case1: parcase1, case2: parcase1, ...}}
        where parcase1 is the parameter par for case1 as a xarray.DataArray.
        """
        
        input_path = data_path
        data = [get_data(input_path, time) for time in dates]
        data_template = self.output_template
        data = np.stack(data, axis = 0)
        output = {}
        # calculate the parameters
        parameters = par_and_cases.keys()
        # we are using a warning catcher here because np.nanmean and np.nanstd will throw a warning if all values are NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # mean
            if 'mean' in parameters:
                mean_data = np.nanmean(data, axis = 0)
                mean = data_template.copy(data = mean_data)
                output['mean'] = {0:mean}
            # std
            if 'std' in parameters:
                std_data = np.nanstd(data, axis = 0)
                std = data_template.copy(data = std_data)
                output['std'] = {0:std}
        return output
    
    def calc_index(self, time,  history: TimeRange, case: dict) -> xr.DataArray:
        # load the data for the index
        input_path_raw = self.output_paths['data']
        input_path = substitute_values(input_path_raw, case['tags'])
        data = get_data(input_path, time)

        # load the parameters
        tag_dict = {'history_start': history.start, 'history_end': history.end}
        tag_dict.update(case['tags'])
        par_path = {par: substitute_values(self.output_paths[par], tag_dict)
                        for par in self.parameters}
        parameters = {par: get_data(par_path[par], time) for par in self.parameters}
        mean = parameters['mean']
        # calculate the index
        if case['options']['type'] == 'empiricalzscore':
            std  = parameters['std']
            anomaly_data = (data - mean) / std
        elif case['options']['type'] == 'percentdelta':
            anomaly_data = (data - mean) / mean * 100
        elif case['options']['type'] == 'absolutedelta':
            anomaly_data = data - mean
        else:
            raise ValueError(f"Unknown type {case['options']['type']} for anomaly index.")
        
        output_template = self.output_template
        anomaly = output_template.copy(data = anomaly_data)
        return anomaly