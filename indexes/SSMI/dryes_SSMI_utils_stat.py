#######################################################################################
# Library
import logging
import os
import json

import numpy as np
from scipy.ndimage import uniform_filter1d
import scipy.stats as stat

#######################################################################################


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

