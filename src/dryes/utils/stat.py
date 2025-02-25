import numpy as np
import scipy.stats as stat
from lmoments3 import distr
from typing import Sequence

## supported distributions
# gamma
# normal
# pearson3
# gev
# beta
# genlog

# Gets parameters of a gamma distribution fitted to the data x
def compute_distr_parameters(x: Sequence[float], distribution: str) -> list[float]:
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

        return fit
    
    elif distribution == 'genlog':
        parnames = ['loc', 'scale', 'k']
        this_distr = distr.glo

    # filter the nans out of the data
    x = x[~np.isnan(x)]

    try:
        fit_dict = this_distr.lmom_fit(x)
    except:
        return [np.nan]*len(parnames)

    fit = [fit_dict[pn] for pn in parnames]
    
    return fit

# Get the p-value of the Kolmogorov-Smirnov test of the data x to a distribution
def get_pval(x: Sequence[float],
             distribution: str,
             fit = np.ndarray) -> float:

    if any(np.isnan(fit)):
        return 0.0

    x = np.array(x)

    if distribution == 'normal':
        distribution = 'norm' # this is the name used by scipy
    elif distribution == 'gev':
        distribution = 'genextreme' # this is the name used by scipy
    elif distribution == 'genlog':
        # TODO: ADD A WARNING AT SOME POINT, this distribution is not available in scipy
        return 1.0

    fit = list(fit)
    _, p_value = stat.kstest(x, distribution, args=fit)
    return p_value

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

        # AT SOME POINT DO ALL THE FITTING WITH lmoments3, it is more consistent, given we calculate the parameters with it.
        if distribution == 'genlog':
            fitted = distr.glo(**pars)
            probVal = fitted.cdf(data)
        else:
            # compute SPI
            probVal = randvar.cdf(data, **pars)

        # correct for the probability of zero, if needed
        if 'prob0' in parameters.keys():
            prob0 = parameters['prob0']
            probVal = prob0 + ((1 - prob0) * probVal)

        probVal = np.where(probVal == 0, corr_extremes, probVal)
        probVal = np.where(probVal == 1, 1 - corr_extremes, probVal)

        return probVal

# Maps a probability value to a normal distribution
def map_prob_to_normal(probVal: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    return stat.norm.ppf(probVal, loc=loc, scale=scale)