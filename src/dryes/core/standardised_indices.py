import numpy as np
from lmoments3 import distr
import scipy.stats as stat

from typing import Sequence

PARAMETERS = {
    'gamma':    ['a', 'loc', 'scale', 'prob0'],
    'normal':   ['loc', 'scale'],
    'pearson3': ['skew', 'loc', 'scale'],
    'gev':      ['c', 'loc', 'scale'],
    'beta':     ['a', 'b'],
    'genlog':   ['loc', 'scale', 'k']
}

def calc_spi(data: np.ndarray, parameters: dict, zero_threshold: float = 0.01) -> np.ndarray:
    """
    Calculate the SPI of the given (precipitation) data.
    Parameters:
        data (np.ndarray): The input precipitation data for which the SPI is to be calculated.
        parameters (dict): A dictionary containing the parameters of the generalised logistic distribution.
                           The expected keys are:
                           - 'a': The shape parameter of the gamma distribution.
                           - 'loc': The location parameter of the gamma distribution.
                           - 'scale': The scale parameter of the gamma distribution.
                           - 'prob0': The probability of zero in the data.
        zero_threshold (float): The threshold below which values are considered as zero.
    Returns:
        np.ndarray: The calculated SPI values.
    """
    return calc_standardised_index(data, 'gamma', parameters, zero_threshold)

def fit_spi(data: np.ndarray, min_n: int = 0, zero_threshold: float = 0.01) -> dict:
    """
    Fits a gamma distribution (for SPI) to the data and returns the parameters.
    The function uses the lmoments3 library to fit the distribution to the data.
    The function will return NaN for all parameters if there are not enough data points to fit the distribution.
    The function will also return NaN for all parameters if the fit fails.

    Parameters:
        data (np.ndarray): A 3D numpy array where the 0-th axis is time, and the remaining axes 
                           represent spatial or other dimensions. The data is assumed to correspond 
                           to cumulative precipitation for a  single period in the reference period
                           (e.g., all January months in the reference period).
        min_n (int): The minimum number of non-nan (and non-zero, for gamma distribution) data points required to fit the distribution.
        zero_threshold (float): The threshold below which values are considered as zero.
    Returns:
        dict: A dictionary containing the parameters of the fitted gamma distribution.
              The parameters are:
              - 'a': The shape parameter of the gamma distribution.
              - 'loc': The location parameter of the gamma distribution.
              - 'scale': The scale parameter of the gamma distribution.
              - 'prob0': The probability of zero in the data.
    """
    return fit_data(data, 'gamma', min_n, zero_threshold)

def calc_spei(data: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Calculate the SPEI of the given (water balance) data.
    Parameters:
        data (np.ndarray): The input water balance data (precipitation-PET) for which the SPEI is to be calculated.
        parameters (dict): A dictionary containing the parameters of the generalised logistic distribution.
                           The expected keys are:
                           - 'loc': The location parameter of the generalised logistic distribution.
                           - 'scale': The scale parameter of the generalised logistic distribution.
                           - 'k': The shape parameter of the generalised logistic distribution.
    Returns:
        np.ndarray: The calculated SPEI values.
    """
    return calc_standardised_index(data, 'genlog', parameters)

def fit_spei(data: np.ndarray, min_n: int = 0):
    """
    Fits a generalised logistic distribution, aka log-logistic (for SPEI) to the data and returns the parameters.
    The function uses the lmoments3 library to fit the distribution to the data.
    The function will return NaN for all parameters if there are not enough data points to fit the distribution.
    The function will also return NaN for all parameters if the fit fails.

    Parameters:
        data (np.ndarray): A 3D numpy array where the 0-th axis is time, and the remaining axes 
                           represent spatial or other dimensions. The data is assumed to correspond 
                           to water balance (i.e. precipitation - PET) for single period in the reference period
                           (e.g., all January months in the reference period).
        min_n (int): The minimum number of non-nan (and non-zero, for gamma distribution) data points required to fit the distribution.
    Returns:
        dict: A dictionary containing the parameters of the fitted gamma distribution.
              The parameters are:
              - 'loc': The location parameter of the generalised logistic distribution.
              - 'scale': The scale parameter of the generalised logistic distribution.
              - 'k': The shape parameter of the generalised logistic distribution.
    """
    return fit_data(data, 'genlog', min_n)

def calc_standardised_index(data: np.ndarray, distribution: str, parameters: dict, zero_threshold: float = 0.01) -> np.ndarray:
    """
    Calculate the fitted standardised anomaly of the given data using the specified method.
    Parameters:
        data (np.ndarray): The input data for which the fitted standardised anomaly is to be calculated.
        distribution (str): The name of the fitted distribution to use for the calculation.
                            One of the following: 'gamma', 'normal', 'pearson3', 'gev', 'beta', 'genlog'.
        parameters (dict): A dictionary containing the parameters of the fitted distribution.
                           Each distribution has its own set of parameters:
                           - 'gamma': ['a', 'loc', 'scale', 'prob0']
                           - 'normal': ['loc', 'scale']
                           - 'pearson3': ['skew', 'loc', 'scale']
                           - 'gev': ['c', 'loc', 'scale']
                           - 'beta': ['a', 'b']
                           - 'genlog': ['loc', 'scale', 'k']
        zero_threshold (float): The threshold below which values are considered as zero (only used for gamma distribution).
    Returns:
        np.ndarray: The calculated fitted standardised anomaly values.
    Raises:
        ValueError: If the specified distribution is unknown or if required parameters are missing.
    """

    if distribution not in PARAMETERS.keys():
        raise ValueError(f"Unknown distribution {distribution}.")

    for par in PARAMETERS[distribution]:
        if par not in parameters.keys():
            raise ValueError(f"Missing parameter {par} for {distribution} distribution.")

    # remove the zeros, if we have a gamma distribution
    if distribution == 'gamma':
        iszero = data <= zero_threshold
        iszero = np.where(np.isnan(data), np.nan, iszero)
        data   = np.where(iszero, np.nan, data)

    # get the probability of the data to be in the fitted distribution
    probVal = get_prob(data, distribution, parameters)

    # map the probability values to a standard normal distribution
    return map_prob_to_normal(probVal)

def fit_data(data: np.ndarray, distribution: str, min_n: int = 0, zero_threshold: float = 0.01) -> dict:

    """
    Fits the specified distribution to the data and returns the parameters.
    The function uses the lmoments3 library to fit the distribution to the data.
    The function will return NaN for all parameters if there are not enough data points to fit the distribution.
    The function will also return NaN for all parameters if the fit fails.

    Parameters:
        data (np.ndarray): A 3D numpy array where the 0-th axis is time, and the remaining axes 
                           represent spatial or other dimensions. The data is assumed to correspond 
                           to a single period in the reference period (e.g., all January months in 
                           the reference period).
        distribution (str): The name of the distribution to fit.
                            One of the following: 'gamma', 'normal', 'pearson3', 'gev', 'beta', 'genlog'.
        min_n (int): The minimum number of non-nan (and non-zero, for gamma distribution) data points required to fit the distribution.
        zero_threshold (float): The threshold below which values are considered as zero (only used for gamma distribution).
    Returns:
        dict: A dictionary containing the parameters of the fitted distribution.
            Each distribution has its own set of parameters:
            - 'gamma': ['a', 'loc', 'scale', 'prob0']
            - 'normal': ['loc', 'scale']
            - 'pearson3': ['skew', 'loc', 'scale']
            - 'gev': ['c', 'loc', 'scale']
            - 'beta': ['a', 'b']
            - 'genlog': ['loc', 'scale', 'k']
    Raises:
        ValueError: If the specified distribution is unknown.
    """

    # fit the distribution to the data, pixel-by-pixel
    distr_parvalues = np.apply_along_axis(fit_data_singlepixel, axis=0, arr=data,
                                          distribution=distribution, min_n=min_n,
                                          zero_threshold=zero_threshold)
    # create the output
    parameters = {}

    # assign names to the parameters
    parnames = PARAMETERS[distribution]
    for ip, par in enumerate(parnames):
        parameters[par] = distr_parvalues[ip]

    return parameters

def get_prob(data: np.ndarray, distribution: str, parameters: dict[str:np.ndarray], corr_extremes = 1e-7) -> np.ndarray:
    """
    Calculates the probability of the data to be in a fitted distribution.
    Parameters:
        data (np.ndarray): The input data for which the probability is to be calculated.
        distribution (str): The name of the distribution to use for the calculation.
                            One of the following: 'gamma', 'normal', 'pearson3', 'gev', 'beta', 'genlog'.
        parameters (dict): A dictionary containing the parameters required for the calculation.
                           Each distribution has its own set of parameters:
                            - 'gamma': ['a', 'loc', 'scale']
                            - 'normal': ['loc', 'scale']
                            - 'pearson3': ['skew', 'loc', 'scale']
                            - 'gev': ['c', 'loc', 'scale']
                            - 'beta': ['a', 'b']
                            - 'genlog': ['loc', 'scale', 'k']
                            Additionally, 'prob0' can be included to correct for the probability of zero.
        corr_extremes (float): A small value to correct extreme probabilities. Defaults to 1e-7.
    Returns:
        np.ndarray: The calculated probability values.
    Raises:
        ValueError: If the specified distribution is unknown.
    """

    if distribution not in PARAMETERS.keys():
        raise ValueError(f"Unknown distribution {distribution}.")

    # extract only the parameters for this distribution
    pars = {k:parameters[k] for k in PARAMETERS[distribution]}

    # get the probability of 0
    if 'prob0' in parameters.keys():
        prob0 = pars.pop('prob0')
    else:
        prob0 = 0

    if distribution == 'genlog':
        # genlog distribution is not in scipy.stats, so we use lmoments3
        probVal = distr.glo.cdf(data, **pars)
    
    else:
        if distribution == 'gamma':
            randvar = stat.gamma
        elif distribution == 'normal':
            randvar = stat.norm
        elif distribution == 'pearson3':
            randvar = stat.pearson3
        elif distribution == 'gev':
            randvar = stat.genextreme
        elif distribution == 'beta':
            randvar = stat.beta

        probVal = randvar.cdf(data, **pars)

    # correct for the probability of zero, if needed
    probVal = prob0 + ((1 - prob0) * probVal)

    probVal = np.where(probVal == 0, corr_extremes, probVal)
    probVal = np.where(probVal == 1, 1 - corr_extremes, probVal)

    return probVal

def fit_data_singlepixel(data: Sequence[float],
                            distribution: str,
                            min_n: int = 0,
                            zero_threshold: float = 0.01) -> list:
    x = np.array(data)
    #if np.nansum(data) > 0: breakpoint()
    if distribution == 'gamma':

        # calculate the probability of zero
        zeros = x <= zero_threshold
        zeros = np.where(np.isnan(x), np.nan, zeros)
        prob0 = np.nanmean(zeros)

        # reassign the zeros to NaN
        x = np.where(zeros, np.nan, x)

        this_distr = distr.gam
    elif distribution == 'normal':
        this_distr = distr.nor
    elif distribution == 'pearson3':
        this_distr = distr.pe3
    elif distribution == 'gev':
        this_distr = distr.gev
    elif distribution == 'genlog':
        this_distr = distr.glo
    elif distribution == 'beta':

        # filter the nans out of the data
        x = x[~np.isnan(x)]

        # get mean and variance
        mean = np.mean(x)
        var = np.var(x)

        # get the parameters
        a = mean * ((mean * (1 - mean)) / var - 1)
        b = (1 - mean) * ((mean * (1 - mean)) / var - 1)

        # return the parameters
        return [a, b] if len(x) >= min_n else [float('nan'), float('nan')]
    else:
        raise ValueError(f"Unknown distribution {distribution}.")

    # filter the nans out of the data
    x = x[~np.isnan(x)]

    # if there are not enough data points, we return NaN for all parameters
    if len(x) < min_n:
        return [float('nan')] * len(PARAMETERS[distribution])

    try:
        parameters = this_distr.lmom_fit(x)
        
        # if we are using the gamma distribution, we need to add the probability of zero
        if distribution == 'gamma':
            parameters['prob0'] = prob0

        return [parameters[k] for k in PARAMETERS[distribution]]
    except:
        # if the fit fails, we return NaN for all parameters
        return [float('nan')] * len(PARAMETERS[distribution])

# Maps a probability value to a normal distribution
def map_prob_to_normal(probVal: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    return stat.norm.ppf(probVal, loc=loc, scale=scale)

# Get the p-value of the Kolmogorov-Smirnov test of the data x to a distribution
def get_pval(data: Sequence[float],
             distribution: str,
             fit: Sequence[float],
             zero_threshold: float = 0.01) -> float:

    if any(np.isnan(fit)):
        return 0.0

    x = np.array(data)

    if distribution == 'normal':
        distribution = 'norm' # this is the name used by scipy
    elif distribution == 'gev':
        distribution = 'genextreme' # this is the name used by scipy
    elif distribution == 'genlog':
        # TODO: ADD A WARNING AT SOME POINT, this distribution is not available in scipy
        return 1.0
    elif distribution == 'gamma':
        x = x[x > zero_threshold]

    fit = list(fit)
    _, p_value = stat.kstest(x, distribution, args=fit)
    return p_value