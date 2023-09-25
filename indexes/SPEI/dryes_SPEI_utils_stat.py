#######################################################################################
# Library
import logging
import os
import json

import numpy as np
from scipy.ndimage import uniform_filter1d
import scipy.stats as stat
from lmoments3 import distr

#######################################################################################


# -------------------------------------------------------------------------------------
# Method to fit a GEV and perform KS test on matrix data
def fit_gev_2d(data, p_value_threshold, flag_ks, mask):

    parameters = np.zeros(shape=(data.shape[0], data.shape[1], 3)) * np.nan  # initialize container for parameters
    invalid_pixels = 0
    for i in range(np.shape(data)[0]):

        for j in range(np.shape(data)[1]):

            if (sum(np.isnan(data[i, j, :])) < np.shape((data))[2]) \
                    & (sum(np.isfinite(data[i, j, :])) >= 4) & (mask[i,j] == 1):

                data1d = data[i, j, :]
                data1d = data1d[np.where(np.isfinite(data1d))]

                try:
                    fit_dict = distr.gev.lmom_fit(data1d)
                    fit = (fit_dict['c'], fit_dict['loc'], fit_dict['scale'])

                    if flag_ks:

                        max_distance, p_value = stat.kstest(data1d, "genextreme", args=fit)
                        if p_value > p_value_threshold:
                            parameters[i, j, 0] = fit[0]
                            parameters[i, j, 1] = fit[1]
                            parameters[i, j, 2] = fit[2]
                        else:
                            invalid_pixels += 1

                    else:
                        parameters[i, j, 0] = fit[0]
                        parameters[i, j, 1] = fit[1]
                        parameters[i, j, 2] = fit[2]

                except:
                    logging.warning(' ==> Fitting failed at row ' + str(i) + ' and  column ' + str(j))
                    invalid_pixels += 1
                    continue

    logging.info(' --> WARNING: ' + str(invalid_pixels) + ' pixels were rejected by KS!')
    return parameters

# -------------------------------------------------------------------------------------

