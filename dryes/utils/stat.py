import numpy as np
import scipy.stats as stat
from lmoments3 import distr
from typing import Iterable

## supported distributions
# gamma
# normal
# pearson3
# gev
# beta

# Gets parameters of a gamma distribution fitted to the data x
def compute_distr_parameters(x: Iterable[float], distribution: str,
                             min_obs: int = 5) -> list[float]:
    x = np.array(x)

    if distribution == 'gamma':
        parnames = ['a', 'loc', 'scale']
        this_distr = distr.gam
    elif distribution == 'normal':
        parnames = ['loc', 'scale']
        this_distr = distr.nor
    elif distribution == 'pearson3':
        parnames = ['skew', 'loc', 'scale']
        this_distr = distr.pe3
    elif distribution == 'gev':
        parnames = ['c', 'loc', 'scale']
        this_distr = distr.gev
    elif distribution == 'beta':
        parnames = ['a', 'b']

        # filter the nans out of the data
        x = x[~np.isnan(x)]

        # get mean and variance
        mean = np.mean(x)
        var = np.var(x)

        # get the parameters
        a = mean * ((mean * (1 - mean)) / var - 1)
        b = (1 - mean) * ((mean * (1 - mean)) / var - 1)

        # return the parameters
        fit = [a, b]

        # assign the parnames to the parameters
        fit = dict(zip(parnames, fit))

        return fit

    # filter the nans out of the data
    x = x[~np.isnan(x)]

    if len(x) < min_obs:
        return [np.nan]*len(parnames)
    try:
        fit_dict = this_distr.lmom_fit(x)
    except:
        return [np.nan]*len(parnames)

    fit = [fit_dict[pn] for pn in parnames]
    
    return fit

# Checks if the p-value of the Kolmogorov-Smirnov test is above the threshold
def check_pval(x: Iterable[float], distribution: str,
               fit = np.ndarray,
               p_val_th: float = None) -> bool:

    if any(np.isnan(fit)):
        return False
    if p_val_th is None:
        return True

    x = np.array(x)

    if distribution == 'normal':
        distribution = 'norm' # this is the name used by scipy
    elif distribution == 'gev':
        distribution = 'genextreme' # this is the name used by scipy

    fit = list(fit)
    _, p_value = stat.kstest(x, distribution, args=fit)
    if p_value < p_val_th:
        return False
    else:
        return True

# Gets the probability of the data x to be in a gamma distribution
def get_prob(data: np.ndarray, distribution: str, 
             parameters: dict[str:np.ndarray],
             corr_extremes = 1e-7) -> np.ndarray:
        
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

        # remove the name of the distribution from the parameters name and only select the ones for this distribution
        pars = {k.replace(f'{distribution}.', ''):v for k,v in parameters.items() if k.startswith(f'{distribution}.')}

        # compute SPI
        probVal = randvar.cdf(data, **pars)
        probVal[probVal == 0] = corr_extremes
        probVal[probVal == 1] = 1 - corr_extremes

        # correct for the probability of zero, if needed
        if 'prob0' in parameters.keys():
            prob0 = parameters['prob0']
            probVal = prob0 + ((1 - prob0) * probVal)

        return probVal

# Maps a probability value to a normal distribution
def map_prob_to_normal(probVal: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    return stat.norm.ppf(probVal, loc=loc, scale=scale)