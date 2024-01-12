import numpy as np
import scipy.stats as stat
from lmoments3 import distr
from typing import Iterable

# Gets parameters of a gamma distribution fitted to the data x
def compute_gamma(x: Iterable[float]) -> tuple[float, float, float]: 
    x = np.array(x)
    dpd = x[np.where(x > 0)]  # only non null values

    if len(dpd) < 4:
        return [np.nan]*3

    try:
        fit_dict = distr.gam.lmom_fit(dpd)
    except:
        return [np.nan]*3

    fit = [fit_dict['a'],fit_dict['loc'],fit_dict['scale']]
    
    return fit

# Checks if the p-value of the Kolmogorov-Smirnov test is above the threshold
def check_pval_gamma(x: Iterable[float], fit = np.ndarray, p_val_th: float = None) -> bool:

    if any(np.isnan(fit)):
        return False

    x = np.array(x)
    dpd = x[np.where(x > 0)]  # only non null values

    fit = list(fit)
    if p_val_th is not None:
        _, p_value = stat.kstest(dpd, "gamma", args=fit)
    if p_value < p_val_th:
        return False

    return True

# Gets the probability of the data x to be in a gamma distribution
def get_prob_gamma(data: np.ndarray, parameters: dict[str:np.ndarray], corr_extremes = 1e-7) -> np.ndarray:
        
        # get the parameters
        a = parameters['gamma.a']
        loc = parameters['gamma.loc']
        scale = parameters['gamma.scale']
        
        # compute SPI
        probVal = stat.gamma.cdf(data, a=a, loc=loc, scale=scale)
        probVal[probVal == 0] = corr_extremes
        probVal[probVal == 1] = 1 - corr_extremes

        if 'prob0' in parameters.keys():
            prob0 = parameters['prob0']
            probVal = prob0 + ((1 - prob0) * probVal)

        return probVal

# Maps a probability value to a normal distribution
def map_prob_to_normal(probVal: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    return stat.norm.ppf(probVal, loc=loc, scale=scale)