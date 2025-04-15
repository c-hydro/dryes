import numpy as np
import warnings

from typing import Optional

from .dryes_index import DRYESIndex
from ..core.standardised_indices import calc_standardised_index, fit_data, get_pval

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
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # fit the distribution to the data
                parameters = fit_data(data, options['distribution'], zero_threshold = options['zero_threshold'])

            # change the names of the keys to match the distribution
            for par in self.distr_par[options['distribution']]:
                pname = par.replace(f'{options["distribution"]}.', '')
                parameters[par] = parameters.pop(f'{pname}')

            # write a parameter for the number of data points used for the fitting
            if options['distribution'] == 'gamma':
                parameters['n'] = np.sum(np.logical_and(np.logical_not(np.isnan(data)), data > options['zero_threshold']), axis=0)
            else:
                parameters['n'] = np.sum(np.logical_not(np.isnan(data)), axis=0)

            # check if we need to calculate the p-values
            if options['pval_check']:
                par_np = np.stack([parameters[par] for par in self.distr_par[options['distribution']]], axis=0)
                to_iterate = np.where(~np.isnan(par_np[0])) # only iterate over the non-nan values of the parameters
                pvals = np.zeros(par_np[0].shape)
                for x,y in zip(*to_iterate):
                    pvals[x,y] = get_pval(data[:,x,y], options['distribution'], par_np[:,x,y], zero_threshold = options['zero_threshold'])

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

        distribution = options['distribution']

        # remove the name of the distribution from the parameters name and only select the ones for this distribution
        pars = {k.replace(f'{distribution}.', ''):v for k,v in parameters.items() if k.startswith(f'{distribution}.')}

        # if we use the gamma distribution, we need to add the probability of zero
        if distribution == 'gamma': pars['prob0'] = parameters['prob0']

        # calculate and return the standardised index
        return calc_standardised_index(data, distribution, pars, options['zero_threshold'])
    
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

    def _check_io_data(self, io_options: dict) -> None:
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

        self._data._template = self.output_template

        self._raw_inputs['PET']._template = self.output_template
        self._raw_inputs['P']._template = self.output_template

        self._data.set_parents({'P': self._raw_inputs['P'], 'PET': self._raw_inputs['PET']}, lambda P, PET: P - PET)

class SSMI(DRYESStandardisedIndex):
    index_name = 'SSMI'
    default_options = {
        'distribution'   : 'beta',
    }