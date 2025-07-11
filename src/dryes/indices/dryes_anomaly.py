import numpy as np
import warnings

from typing import Optional

from .dryes_index import DRYESIndex
from ..core.anomalies import calc_anomaly, calc_anomaly_parameters

class DRYESAnomaly(DRYESIndex):
    index_name = 'anomaly'

    default_options = {
        'type'         : 'empiricalzscore',
        'min_reference': 1,
        'min_std'      : 0.01,

        # derived options
        'get_std'      : True
    }

    option_cases = {
        'parameters_1' : ['get_std'],
        'parameters_2' : ['min_reference'],
        'index_1'      : ['type'],
        'index_2'      : ['min_std']
    }

    @property
    def parameters(self):
        all_types = set(c.options['type'] for c in self.cases[-1].values())
        return ['mean', 'std'] if 'empiricalzscore' in all_types else ['mean']

    def __init__(self,
                 io_options: dict,
                 index_options: dict = {},
                 run_options: Optional[dict] = None) -> None:
        
        get_std = False
        type_options = index_options.get('type')
        if type_options is not None:
            if type_options == 'empiricalzscore':
                get_std = True
            elif isinstance(type_options, dict) and 'empiricalzscore' in type_options.values():
                get_std = True
            index_options['get_std'] = get_std
        
        super().__init__(io_options, index_options, run_options)

    def calc_parameters(self,
                        data: np.ndarray|dict[str, np.ndarray], options: dict, 
                        step = 1, **kwargs) -> dict[str, np.ndarray]:
        """
        Calculates the parameters for the index.
        """
        
        parameters = {}

        # first step in parameter calculation is the calculation of the mean and eventually std
        if step == 1:
            # we are using a warning catcher here because np.nanmean and np.nanstd will throw a warning if all values are NaN
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                parameters = calc_anomaly_parameters(data, get_std = options['get_std'])

            # write a parameter for the number of data points used for the fitting
            parameters['n'] = np.sum(~np.isnan(data), axis=0)

        # second step is the filtering based on the the number of nans
        # here the input (data) are the fitted parameters (including the pvalues)
        elif step == 2:
            parameters: dict = data

            mask = np.where(parameters.pop('n') > options['min_reference'], True, False)
            
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

        if step == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                anomaly_data = calc_anomaly(data, parameters, options['type'])

        elif step == 2:
            anomaly_data = data

            if options['type'] == 'empiricalzscore':
                mask = np.where(parameters['std'] < options['min_std'], True, False)
                anomaly_data = np.where(mask, np.nan, anomaly_data)

            elif options['type'] == 'percentdelta':
                mask = np.where(parameters['mean'] < options['min_std'], True, False)
                anomaly_data = np.where(mask, np.nan, anomaly_data)

        return anomaly_data