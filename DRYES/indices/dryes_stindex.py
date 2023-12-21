from datetime import datetime
import numpy as np
import xarray as xr
import warnings

from typing import List, Optional

from .dryes_index import DRYESIndex

from ..time_aggregation import aggregation_functions as agg
from ..lib.time import TimeRange
from ..lib.parse import substitute_values
from ..lib.io import get_data
from ..lib.stat import compute_gamma, check_pval_gamma, get_prob_gamma, map_prob_to_normal


class DRYESStandardisedIndex(DRYESIndex):
    index_name = 'standardised index'
    default_options = {
        'agg_fn'         : {'Agg1': agg.average_of_window(1, 'months')},
        'distribution'   : 'gamma',
        'pval_threshold' : None
    }
    
    @property
    def parameters(self):
        opt_cases = self.cases['opt']
        all_distr = set(case['options']['distribution'] for case in opt_cases)
        all_parameters = ['prob0']
        for distribution in all_distr:
            if distribution == 'gamma':
                all_parameters += ['gamma.a', 'gamma.loc', 'gamma.scale']
            # elif distribution == 'normal':
            #     all_parameters += ['normal.loc', 'normal.scale']
            else:
                raise ValueError(f"Unknown distribution {distribution}.")
        return all_parameters

    def calc_parameters(self, dates: List[datetime],
                        data_path: str,
                        par_and_cases: dict[str:List[int]],
                        reference: Optional[TimeRange]=None) -> dict[str:dict[int:xr.DataArray]]:
        """
        Calculates the parameters for the standardised index (i.e. the distribution fitting).
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
        # remove the 1-dimensional band dimension
        #data = data.squeeze()
        output = {}
        # calculate the parameters
        parameters = par_and_cases.keys()
        # we are using a warning catcher here because np.nanmean and np.nanstd will throw a warning if all values are NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # prob0 -> this is for all distributions
            if 'prob0' in parameters:
                iszero = data == 0
                # reassign NaNs
                iszero = np.where(np.isnan(data), np.nan, iszero)
                prob0_data = np.nanmean(iszero, axis = 0)
                prob0 = data_template.copy(data = prob0_data)
                output['prob0'] = {0:prob0} # -> we save this as case 0 because it is the same for all cases
            
            # gamma distribution
            gamma_pars = [p for p in parameters if 'gamma' in p]
            if len(gamma_pars) > 0:
                gamma_parameters = np.apply_along_axis(compute_gamma, axis = 0, arr = data)
                for case in par_and_cases[gamma_pars[0]]:
                    these_pars = gamma_parameters.copy()
                    this_p_thr = self.cases['opt'][case]['options']['pval_threshold']
                    to_iterate = np.where(~np.isnan(gamma_parameters[0]))
                    for b,x,y in zip(*to_iterate):
                        if not check_pval_gamma(data[:,b,x,y], gamma_parameters[:,b,x,y], p_val_th = this_p_thr):
                            these_pars[:,b,x,y] = np.nan

                    for i, par in enumerate(gamma_pars):
                        this_par_data = these_pars[i]
                        this_par = data_template.copy(data = this_par_data)
                        output[par] = {case: this_par}
                        
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
        # calculate the index
        if case['options']['distribution'] == 'gamma':
            probVal = get_prob_gamma(data, parameters)
        # elif case['options']['distribution'] == 'normal':
        #     probVal = get_prob_normal(data, parameters)
        else:
            raise ValueError(f"Unknown distribution {case['options']['distribution']}.")
        
        stindex_data = map_prob_to_normal(probVal)

        output_template = self.output_template
        anomaly = output_template.copy(data = stindex_data)
        return anomaly