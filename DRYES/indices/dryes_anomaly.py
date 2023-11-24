from datetime import datetime
import numpy as np
import warnings
from typing import List

from .dryes_index import DRYESIndex

from ..time_aggregation import aggregation_functions as agg
from ..lib.time import TimeRange, ntimesteps_to_md
from ..lib.parse import substitute_values
from ..lib.log import log
from ..lib.io import get_data, check_data, save_dataarray_to_geotiff


class DRYESAnomaly(DRYESIndex):

    default_options = {
        'agg_fn' : {'Agg1': agg.average_of_window(1, 'months')},
        'type'   : 'empiricalzscore'
    }
    
    @property
    def parameters(self):
        opt_cases = self.cases['opt']
        all_types = [case['options']['type'] for case in opt_cases]
        return ['mean', 'std'] if 'empiricalzscore' in all_types else ['mean']

    def calc_parameters(self, dates: List[datetime], parameters: List[str]) -> dict:
        input_path = self.input_variable.path
        data = [get_data(input_path, time) for time in dates]
        data_template = data[0].copy()
        data = np.stack(data, axis = 0)
        output = {}
        # calculate the parameters
        # we are using a warning catcher here because np.nanmean and np.nanstd will throw a warning if all values are NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # mean
            if 'mean' in parameters:
                mean_data = np.nanmean(data, axis = 0)
                mean = data_template.copy(data = mean_data)
                output['mean'] = mean
            # std
            if 'std' in parameters:
                std_data = np.nanstd(data, axis = 0)
                std = data_template.copy(data = std_data)
                output['std'] = std
        return output