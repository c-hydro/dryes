from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import warnings

from typing import List

from .dryes_index import DRYESIndex

from ..io import IOHandler
from ..time_aggregation import aggregation_functions as agg

from ..utils.time import TimeRange, get_md_dates
from ..utils.stat import compute_distr_parameters, check_pval, get_prob, map_prob_to_normal


class DRYESStandardisedIndex(DRYESIndex):
    """
    This class implements a standardised index, that is an anomaly index, fitted to a distribution and standardised to a normal distribution.
    It is used for the SPI (Standardised Precipitation Index) and the SPEI (Standardised Precipitation Evapotranspiration Index).
    """
    index_name = 'standardised index'

    default_options = {
        'agg_fn'         : {'Agg1': agg.average_of_window(1, 'months')},
        'distribution'   : 'normal',
        'pval_threshold' : None,
        'min_reference'  : 5,
        'zero_threshold' : 0.0001,
    }
    
    distr_par = {
        'gamma':    ['gamma.a', 'gamma.loc', 'gamma.scale'],
        'normal':   ['normal.loc', 'normal.scale'],
        'pearson3': ['pearson3.skew', 'pearson3.loc', 'pearson3.scale'],
        'gev':      ['gev.c', 'gev.loc', 'gev.scale']
    }

    # @property
    # def positive_only(self):
    #     # this is by default False, unless, we request a gamma distribution
    #     if 'gamma' in self.distributions:
    #         return True
    #     else:
    #         return False

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
        parameters = []
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
        all_dates  = get_md_dates(history_years, time.month, time.day)
        data_dates = [date for date in all_dates if date >= history.start and date <= history.end]

        data_ = [variable.get_data(time) for time in data_dates]
        data = np.stack(data_, axis = 0)

        output = {}
        # calculate the parameters
        parameters = par_and_cases.keys()
        cases_and_pars = {case: [par for par in parameters if case in par_and_cases[par]] for case in range(len(self.cases['opt']))}

        # we are using a warning catcher here because np.nanmean and np.nanstd will throw a warning if all values are NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for case in self.cases['opt']:
                i = case['id']
                if 'prob0' in cases_and_pars[i]:
                    iszero = data <= case['options']['zero_threshold']
                    # reassign NaNs
                    iszero = np.where(np.isnan(data), np.nan, iszero)
                    prob0_data = np.nanmean(iszero, axis = 0)
                    if 'prob0' not in output:
                        output['prob0'] = {i: prob0_data}
                    else:
                        output['prob0'][i] = prob0_data

                # if there is any other parameter, these are the distribution parameters
                if any(par != 'prob0' for par in cases_and_pars[i] ):
                    distr = case['options']['distribution']
                    this_data = data.copy()
                    if distr == 'gamma':
                        this_data = np.where(this_data <= case['options']['zero_threshold'], np.nan, this_data)
                        
                    distr_parnames  = self.distr_par[distr]
                    distr_parvalues = np.apply_along_axis(compute_distr_parameters, axis=0, arr=this_data,
                                                          distribution=distr,
                                                          min_obs = case['options']['min_reference'],)
                    
                    this_p_thr = case['options']['pval_threshold']
                    to_iterate = np.where(~np.isnan(distr_parvalues[0])) # only iterate over the non-nan values of the parameters
                    for b,x,y in zip(*to_iterate):
                        if not check_pval(data[:,b,x,y], distr, distr_parvalues[:,b,x,y], p_val_th = this_p_thr):
                            distr_parvalues[:,b,x,y] = np.nan

                    for ip, par in enumerate(distr_parnames):
                        this_par_data = distr_parvalues[ip]
                        if par not in output:
                            output[par] = {i: this_par_data}
                        else:
                            output[par][i] = this_par_data
        
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
        pars_to_get  = self.distr_par[distribution].copy()
        if 'prob0' in self._parameters:
            pars_to_get += ['prob0']
        par_time = time if time.month != 2 or time.day != 29 else time - timedelta(days = 1)
        parameters = {par: p.get_data(par_time, **case['tags']) for par, p in self._parameters.items() if par in pars_to_get}

        # calculate the index
        probVal = get_prob(data, distribution, parameters)

        stindex_data = map_prob_to_normal(probVal)

        if hasattr(data, 'attrs'):
            stindex_data.attrs = data.attrs
            
        return stindex_data
    
class SPI(DRYESStandardisedIndex):
    index_name = 'SPI (Standardised Precipitaion Index)'
    positive_only = True
    default_options = {
        'agg_fn'         : {'Agg1': agg.average_of_window(1, 'months')},
        'distribution'   : 'gamma',
        'pval_threshold' : None,
        'min_reference'  : 5,
        'zero_threshold' : 0.0001,
    }

    @property
    def parameters(self):
        return super().parameters + ['prob0']

class SPEI(DRYESStandardisedIndex):
    index_name = 'SPEI (Standardised Precipitaion Evapotranspiration Index)'
    positive_only = False
    default_options = {
        'agg_fn'         : {'Agg1': agg.average_of_window(1, 'months')},
        'distribution'   : 'gev',
        'pval_threshold' : None,
        'min_reference'  : 5,
    }

    #TODO: Should we assume that the input data is precipitation minus evapotranspiration? (and the rest is included in DAM)
    #      Or should we allow the user to specify two variables, one for precipitation and one for evapotranspiration?