from datetime import datetime
import numpy as np
import xarray as xr
import warnings

from typing import List

from .dryes_index import DRYESIndex

from ..io import IOHandler
from ..time_aggregation import aggregation_functions as agg

from ..utils.time import TimeRange
from ..utils.stat import compute_distr_parameters, check_pval, get_prob, map_prob_to_normal


class DRYESStandardisedIndex(DRYESIndex):
    index_name = 'standardised index'
    default_options = {
        'agg_fn'         : {'Agg1': agg.average_of_window(1, 'months')},
        'distribution'   : 'gamma',
        'pval_threshold' : None
    }
    
    distr_par = {
        'gamma':    ['gamma.a', 'gamma.loc', 'gamma.scale'],
        'normal':   ['normal.loc', 'normal.scale'],
        'pearson3': ['pearson3.skew', 'pearson3.loc', 'pearson3.scale']
    }

    @property
    def distributions(self):
        opt_cases = self.cases['opt']
        all_distr = set(case['options']['distribution'] for case in opt_cases)
        for distribution in all_distr:
            if distribution not in self.distr_par:
                raise ValueError(f"Unknown distribution {distribution}.")
        return all_distr

    @property
    def parameters(self):
        all_distr = self.distributions
        parameters = ['prob0']
        for distribution in all_distr:
            parameters += [par for par in self.distr_par[distribution]]
        return parameters

    def calc_parameters(self,
                        time: datetime,
                        variable: IOHandler,
                        history: TimeRange,
                        par_and_cases: dict[str:List[int]]) -> dict[str:dict[int:xr.DataArray]]:
        """
        time, variable, history, par_cases
        Calculates the parameters for the index.
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
            # prob0 -> this is for all distributions
            if 'prob0' in parameters:
                iszero = data == 0
                # reassign NaNs
                iszero = np.where(np.isnan(data), np.nan, iszero)
                prob0_data = np.nanmean(iszero, axis = 0)
                output['prob0'] = {0:prob0_data} # -> we save this as case 0 because it is the same for all cases
            
            #distribution parameters
            for distr in self.distributions:
                # fit the distribution to calculate the parameters
                distr_parnames  = self.distr_par[distr]
                distr_parvalues = np.apply_along_axis(compute_distr_parameters, axis=0, arr=data, distribution=distr, positive_only = True)

                # check the p-value of the fit, this needs to be done for each case, as the p-value threshold can be different
                distr_cases = par_and_cases[distr_parnames[0]] # we use the first parameter name to get the cases
                for case in distr_cases:
                    these_pars = distr_parvalues.copy()
                    this_p_thr = self.cases['opt'][case]['options']['pval_threshold']
                    to_iterate = np.where(~np.isnan(these_pars[0])) # only iterate over the non-nan values of the parameters
                    for b,x,y in zip(*to_iterate):
                        if not check_pval(data[:,b,x,y], distr, these_pars[:,b,x,y], p_val_th = this_p_thr):
                            these_pars[:,b,x,y] = np.nan

                    for i, par in enumerate(distr_parnames):
                        this_par_data = these_pars[i]
                        if par not in output:
                            output[par] = {case: this_par_data}
                        else:
                            output[par][case] = this_par_data
                        
        return output
    
    # this is run for a single case, making things a lot easier
    def calc_index(self, time,  history: TimeRange, case: dict) -> xr.DataArray:
        # load the data for the index
        data = self._data.get_data(time, **case['tags'])

        # load the parameters
        if 'history_start' not in case['tags']:
            case['tags']['history_start'] = history.start
        if 'history_end' not in case['tags']:
            case['tags']['history_end'] = history.end

        distribution = case['options']['distribution']
        pars_to_get  = self.distr_par[distribution] + ['prob0']
        parameters   = {par: p.get_data(time, **case['tags']) for par, p in self._parameters.items() if par in pars_to_get}

        # calculate the index
        probVal = get_prob(data, distribution, parameters)

        stindex_data = map_prob_to_normal(probVal)
        return stindex_data