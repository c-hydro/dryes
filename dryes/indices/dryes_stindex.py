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
        'gev':      ['gev.c', 'gev.loc', 'gev.scale'],
        'beta':     ['beta.a', 'beta.b'],
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
    
    # this is run for a single case, making things a lot easier
    def calc_index(self, time:TimeStep,  history: TimeRange, case: dict) -> tuple[np.ndarray, dict]:
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

        distribution = case['options']['distribution']
        pars_to_get  = self.distr_par[distribution].copy()
        if 'prob0' in self._parameters:
            pars_to_get += ['prob0']
        #par_time = time if time.month != 2 or time.day != 29 else time - timedelta(days = 1)
        parameters = {par: p.get_data(time, **case['tags']) for par, p in self._parameters.items() if par in pars_to_get}

        # calculate the index
        probVal = get_prob(data, distribution, parameters)

        stindex_data = map_prob_to_normal(probVal)

        if hasattr(data, 'attrs'):
            index_info = data.attrs
        else:
            index_info = {}

        if hasattr(list(parameters.values())[0], 'attrs'):
            index_info.update(list(parameters.values())[0].attrs)

        # add the parents to the metadata
        parents = parameters
        parameters.update({'data': data})
        index_info['parents'] = parents

        return stindex_data, index_info
    
class SPI(DRYESStandardisedIndex):
    index_name = 'SPI'
    positive_only = True
    default_options = {
        'distribution'   : 'gamma',
    }

    @property
    def parameters(self):
        return super().parameters + ['prob0']

class SPEI(DRYESStandardisedIndex):
    index_name = 'SPEI'
    positive_only = False
    default_options = {
        'distribution'   : 'pearson3',
    }

    def _check_io_options(self, io_options: dict) -> None:
        # for the SPEI, we need an additional step, we also need to check that we have both P_raw and PET_raw!

        if 'PET_raw' not in io_options and 'PET' not in io_options:
            raise ValueError('Either PET or PET_raw must be specified.')
        elif 'P_raw' not in io_options and 'P' not in io_options:
            raise ValueError('Either P or P_raw must be specified.')
        
        self._raw_inputs = {'P': io_options['P_raw'], 'PET': io_options['PET_raw']}
        template = self._raw_inputs['P'].get_template()
        for raw_input in self._raw_inputs.values():
            raw_input.set_template(template)

        if 'data_raw' not in io_options:
            raise ValueError('data_raw must be specified.')
        
        self._raw_data = io_options['data_raw']
        self._raw_data.set_template(template)
        self._raw_data.set_parents({'P': self._raw_inputs['P'], 'PET': self._raw_inputs['PET']}, lambda P, PET: P - PET)

        super()._check_io_options(io_options)

class SSMI(DRYESStandardisedIndex):
    index_name = 'SSMI'
    positive_only = True
    default_options = {
        'distribution'   : 'beta',
    }