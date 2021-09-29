"""
Class Features

Name:          lib_dryes_downloader_hsaf_add_variable_nc
Author(s):     Francesco Avanzi (francesco.avanzi@cimafoundation.org), Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210929'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import os

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
#######################################################################################


# -------------------------------------------------------------------------------------
# Method to compute weighted mean
def compute_weighted_mean(dset, settings_weighted_mean):

    for index, layer in enumerate(settings_weighted_mean['original_variables']):
        layer_da = dset[layer]
        if index == 0:
            weighted_mean = layer_da.values*settings_weighted_mean['weights'][index]
        else:
            weighted_mean = weighted_mean + layer_da.values*settings_weighted_mean['weights'][index]

    weighted_mean_da = xr.DataArray(weighted_mean,
                            dims=layer_da.dims,
                                    coords=layer_da.coords)
    dset[settings_weighted_mean["var_name"]] = weighted_mean_da

    return dset

# -------------------------------------------------------------------------------------
