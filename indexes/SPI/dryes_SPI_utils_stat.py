#######################################################################################
# Library
import logging


import numpy as np
from scipy.ndimage import uniform_filter1d
import scipy.stats as stat
from lmoments3 import distr

#######################################################################################

# -------------------------------------------------------------------------------------
# Method to get gamma parameters
def compute_gamma(x, par, p_val_th):

    x = np.array(x)
    dpd = x[np.where(x > 0)]  # only non null values
    zeros = x[x == 0]

    if len(dpd) < 4:
        return np.nan

    try:
        fit_dict = distr.gam.lmom_fit(dpd)
    except:
        return np.nan

    fit = (fit_dict['a'],fit_dict['loc'],fit_dict['scale'])

    zero_p = len(zeros) / len(x)
    fit_dict.update({"zero_prob": zero_p})

    if p_val_th is not None:

        max_distance, p_value = stat.kstest(dpd, "gamma", args=fit)
        if p_value < p_val_th:
            return np.nan
            # print("Kolmogorov-Smirnov test for goodness of fit: "+str(round(p_value*100))+"%, max distance: "+str(max_distance))

    return fit_dict[par]

# -------------------------------------------------------------------------------------
# Method to perform KS test for the beta distribution on 2d arrays
def kstest_2d_beta(a, parameters,p_value_threshold):
    invalid_pixels = 0
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):

            if sum(np.isnan(a[i, j, :])) < np.shape((a))[2]:

                a1d = a[i, j, :]
                apd = a1d[np.where(a1d > 0)]  # only positive values

                try:
                    fit = (parameters[i, j,0], parameters[i, j,1], parameters[i,j,2], parameters[i,j,3])
                    max_distance, p_value = stat.kstest(apd, "beta", args=fit)

                except:
                    logging.warning(' ==> Fitting failed at row ' + str(i) + ' and  column ' + str(j))
                    continue

                if p_value < p_value_threshold:
                    invalid_pixels += 1
                    parameters[i, j, :] = np.nan

    logging.info(' --> WARNING: ' + str(invalid_pixels) + ' pixels were rejected by KS!')
    return parameters

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to perform KS test for the beta distribution on 2d arrays
def kstest_2d_gamma(a, parameters, p_value_threshold):
    invalid_pixels = 0
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):

            if sum(np.isnan(a[i, j, :])) < np.shape((a))[2]:

                a1d = a[i, j, :]
                apd = a1d[np.where(a1d > 0)]  # only positive values

                try:
                    fit = (parameters[i, j,0], parameters[i, j,1], parameters[i,j,2])
                    max_distance, p_value = stat.kstest(apd, "gamma", args=fit)

                except:
                    logging.warning(' ==> Fitting failed at row ' + str(i) + ' and  column ' + str(j))
                    continue

                if p_value < p_value_threshold:
                    invalid_pixels += 1
                    parameters[i, j, :] = np.nan

    logging.info(' --> WARNING: ' + str(invalid_pixels) + ' pixels were rejected by KS!')
    return parameters

# -------------------------------------------------------------------------------------
