import numpy as np
import warnings

from typing import Optional

from .dryes_index import DRYESIndex
from ..utils.stat import compute_distr_parameters, get_pval, get_prob, map_prob_to_normal

class DRYESStandardisedIndex(DRYESIndex):
    """
    This class implements a standardised index, that is an anomaly index, fitted to a distribution and standardised to a normal distribution.
    It is used for the SPI (Standardised Precipitation Index) and the SPEI (Standardised Precipitation Evapotranspiration Index).
    """
    index_name = 'standardised index'

    default_options = {
        'distribution'   : 'normal',
        'pval_threshold' : None,
        'min_reference'  : 5,
        'zero_threshold' : 0.0001,
        
        # derived options
        'pval_check'     : False
    }

    option_cases = {
        'parameters_1' : ['distribution', 'zero_threshold', 'pval_check'],
        'parameters_2' : ['pval_threshold', 'min_reference'],
        'index'        : ['zero_threshold'],
    }
    
    distr_par = {
        'gamma':    ['gamma.a', 'gamma.loc', 'gamma.scale'],
        'normal':   ['normal.loc', 'normal.scale'],
        'pearson3': ['pearson3.skew', 'pearson3.loc', 'pearson3.scale'],
        'gev':      ['gev.c', 'gev.loc', 'gev.scale'],
        'beta':     ['beta.a', 'beta.b'],
        'genlog':   ['genlog.loc', 'genlog.scale', 'genlog.k'],
    }

    @property
    def distributions(self):
        all_distr = set(c.options['distribution'] for c in self.cases[-1].values())
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

    def __init__(self,
                 io_options: dict,
                 index_options: dict = {},
                 run_options: Optional[dict] = None) -> None:
        
        if index_options.get('pval_threshold') is not None:
            index_options['pval_check'] = True
        else:
            index_options['pval_check'] = False
        
        super().__init__(io_options, index_options, run_options)

    def calc_parameters(self,
                        data: np.ndarray|dict[str, np.ndarray], options: dict, 
                        step = 1, **kwargs) -> dict[str, np.ndarray]:
        """
        Calculates the parameters for the index.
        """

        parameters = {}

        # first step in parameter calculation is the fitting of the distribution
        if step == 1:
            # if we use the gamma distribution, we need the data to be positive
            if options['distribution'] == 'gamma':
                iszero = data <= options['zero_threshold']
                # reassign NaNs
                iszero = np.where(np.isnan(data), np.nan, iszero)
                # we are using a warning catcher here because np.nanmean and np.nanstd will throw a warning if all values are NaN
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    prob0 = np.nanmean(iszero, axis = 0)
                parameters['prob0'] = prob0
                data = np.where(iszero, np.nan, data)

            # then we fit the distribution to the data
            distr_parvalues = np.apply_along_axis(compute_distr_parameters, axis=0, arr=data,
                                        distribution=options['distribution'])
            
            # assign names to the parameters
            parnames = self.distr_par[options['distribution']]
            for ip, par in enumerate(parnames):
                parameters[par] = distr_parvalues[ip]

            # write a parameter for the number of data points used for the fitting
            parameters['n'] = np.sum(~np.isnan(data), axis=0)

            # check if we need to calculate the p-values
            if options['pval_check']:
                to_iterate = np.where(~np.isnan(distr_parvalues[0])) # only iterate over the non-nan values of the parameters
                pvals = np.zeros(distr_parvalues[0].shape)
                for x,y in zip(*to_iterate):
                    pvals[x,y] = get_pval(data[:,x,y], options['distribution'], distr_parvalues[:,x,y])

                parameters['pval'] = pvals
        
        # second step is the filtering based on the p-values and the number of nans
        # here the input (data) are the fitted parameters (including the pvalues)
        elif step == 2:
            parameters: dict = data

            mask = np.where(parameters.pop('n') > options['min_reference'], True, False)
            if options['pval_threshold'] is not None:
                mask = np.logical_and(mask, parameters.pop('pval') > options['pval_threshold'])

            for parname, parvalue in parameters.items():
                parameters[parname] = np.where(mask, parvalue, np.nan)

        return parameters
    
    def calc_index(self,
                   data: np.ndarray, parameters: dict[str, np.ndarray],
                   options: dict, step = 1, **kwargs) -> np.ndarray:
        """
        Calculates the index for the given time and reference period (history).
        Returns the index as a numpy.ndarray and a dictionary of metadata, if any.
        """

        # get the distribution from the options
        distribution = options['distribution']

        if distribution == 'gamma':
            iszero = data <= options['zero_threshold']
            iszero = np.where(np.isnan(data), np.nan, iszero)
            data   = np.where(iszero, np.nan, data)

        # calculate the index
        # we are using a warning catcher here because we will get warnings if there are NaN values in the data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            probVal = get_prob(data, distribution, parameters)
            index   = map_prob_to_normal(probVal)

        return index
    
class SPI(DRYESStandardisedIndex):
    index_name = 'SPI'
    default_options = {
        'distribution'   : 'gamma',
    }

    @property
    def parameters(self):
        return super().parameters + ['prob0']

class SPEI(DRYESStandardisedIndex):
    index_name = 'SPEI'
    default_options = {
        'distribution'   : 'pearson3',
    }

    def _check_io_options(self, io_options: dict) -> None:
        # for the SPEI, we need an additional step, we also need to check that we have both P_raw and PET_raw!

        self._raw_inputs = {'P'  : io_options.get('data_P', None), 
                            'PET': io_options.get('data_PET', None)}
        if any(v is None for v in self._raw_inputs.values()):
            raise ValueError('Both P and PET must be specified.')
        
        # ensure we have the same templates for everything
        self._raw_inputs['PET']._template = self._raw_inputs['P']._template
        self.output_template = self._raw_inputs['P']._template
        
        self._data = io_options.get('data', None)
        if self._data is None: 
            from d3tools.data.memory_dataset import MemoryDataset
            key_pattern = self._raw_inputs['P'].key_pattern.replace('P', 'PminusPET')
            self._data = MemoryDataset(key_pattern)

        self.output_template = self._data._template

        self._raw_inputs['PET']._template = self.output_template
        self._raw_inputs['P']._template = self.output_template

        self._data.set_parents({'P': self._raw_inputs['P'], 'PET': self._raw_inputs['PET']}, lambda P, PET: P - PET)

class SSMI(DRYESStandardisedIndex):
    index_name = 'SSMI'
    default_options = {
        'distribution'   : 'beta',
    }