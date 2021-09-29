"""
Library Features:

Name:          lib_dryes_downloader_io
Author(s):     Francesco Avanzi (francesco.avanzi@cimafoundation.org), Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210929'
Version:       '1.0.0'
"""
#################################################################################
# Library
import os

import numpy as np
from osgeo import gdal, gdalconst
import logging
import matplotlib.pyplot as plt
#################################################################################

# -------------------------------------------------------------------------------------
# Method to write file tiff
def write_file_tiff(file_name, file_data, file_wide, file_high, file_geotrans, file_proj):
    dset_mask = gdal.GetDriverByName('GTiff').Create(file_name, file_wide, file_high, 1,
                                                          gdalconst.GDT_Float32)
    file_data = np.squeeze(file_data)
    dset_mask.SetProjection(file_proj)
    dset_mask.SetGeoTransform(file_geotrans)
    dset_mask.GetRasterBand(1).WriteArray(file_data)
    del dset_mask
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read tiff file
def read_file_tiff(file_name_tiff):
    if os.path.exists(file_name_tiff):
        dset_tiff = gdal.Open(file_name_tiff, gdalconst.GA_ReadOnly)
        dset_proj = dset_tiff.GetProjection()
        dset_geotrans = dset_tiff.GetGeoTransform()
        dset_data = dset_tiff.ReadAsArray()
        dset_band = dset_tiff.GetRasterBand(1)
    else:
        logging.error(' ===> Tiff file ' + file_name_tiff + ' not found')
        raise IOError('Tiff file location or name is wrong')

    return dset_tiff, dset_proj, dset_geotrans, dset_data
# -------------------------------------------------------------------------------------
