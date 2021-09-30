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
import rasterio
from affine import Affine
from collections import namedtuple
#################################################################################

# -------------------------------------------------------------------------------------
# Method to write file tiff
def write_file_tiff(file_name, file_data, file_wide, file_high, file_geotrans, file_proj,
                   file_metadata=None,
                   file_format=gdalconst.GDT_Float32):

    if not isinstance(file_data, list):
        file_data = [file_data]

    if file_metadata is None:
        file_metadata = {'description_field': 'data'}
    if not isinstance(file_metadata, list):
        file_metadata = [file_metadata] * file_data.__len__()

    if isinstance(file_geotrans, Affine):
        file_geotrans = file_geotrans.to_gdal()

    file_crs = rasterio.crs.CRS.from_string(file_proj)
    file_wkt = file_crs.to_wkt()

    file_n = file_data.__len__()
    dset_handle = gdal.GetDriverByName('GTiff').Create(file_name, file_wide, file_high, file_n, file_format,
                                                       options=['COMPRESS=DEFLATE'])
    dset_handle.SetGeoTransform(file_geotrans)
    dset_handle.SetProjection(file_wkt)

    for file_id, (file_data_step, file_metadata_step) in enumerate(zip(file_data, file_metadata)):
        file_data_step = np.squeeze(file_data_step)
        dset_handle.GetRasterBand(file_id + 1).WriteArray(file_data_step)
        dset_handle.GetRasterBand(file_id + 1).SetMetadata(file_metadata_step)
    del dset_handle
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
