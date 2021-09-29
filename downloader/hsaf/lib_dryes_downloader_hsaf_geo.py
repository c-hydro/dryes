"""
Library Features:

Name:          lib_dryes_downloader_geo
Author(s):     Francesco Avanzi (francesco.avanzi@cimafoundation.org), Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210929'
Version:       '1.0.0'
"""
#################################################################################
# Library
import os
import logging
from osgeo import gdal, gdalconst
import numpy as np
import rasterio
import matplotlib.pylab as plt

from lib_dryes_downloader_hsaf_generic import create_darray_2d
#################################################################################

logging.getLogger("rasterio").setLevel(logging.WARNING)

# -------------------------------------------------------------------------------------
# Method to read tiff file
def reproject_file_tiff(file_name_in, file_name_out,
                        file_wide_out, file_high_out, file_geotrans_out, file_proj_out):
    dset_tiff_out = gdal.GetDriverByName('GTiff').Create(
        file_name_out, file_wide_out, file_high_out, 1, gdalconst.GDT_Float32)
    dset_tiff_out.SetGeoTransform(file_geotrans_out)
    dset_tiff_out.SetProjection(file_proj_out)

    dset_tiff_in = gdal.Open(file_name_in, gdalconst.GA_ReadOnly)
    dset_proj_in = dset_tiff_in.GetProjection()
    dset_geotrans_in = dset_tiff_in.GetGeoTransform()
    dset_data_in = dset_tiff_in.ReadAsArray()
    dset_band_in = dset_tiff_in.GetRasterBand(1)

    # Reproject from input file to output file set with out information
    gdal.ReprojectImage(dset_tiff_in, dset_tiff_out, dset_proj_in, file_proj_out,
                        gdalconst.GRA_NearestNeighbour)
    return dset_tiff_out
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to get a raster file
def read_file_raster(file_name, file_proj='epsg:4326', var_name='land',
                     coord_name_x='Longitude', coord_name_y='Latitude',
                     dim_name_x='Longitude', dim_name_y='Latitude', no_data_default=-9999.0):

    if os.path.exists(file_name):
        if (file_name.endswith('.txt') or file_name.endswith('.asc')) or file_name.endswith('.tif'):

            crs = rasterio.crs.CRS({"init": file_proj})
            with rasterio.open(file_name, mode='r+') as dset:
                dset.crs = crs
                bounds = dset.bounds
                no_data = dset.nodata
                res = dset.res
                transform = dset.transform
                data = dset.read()
                proj = dset.crs.wkt
                values = data[0, :, :]

            if (no_data is None) or (np.isnan(no_data)):
                no_data = no_data_default

            decimal_round = 7

            center_right = bounds.right - (res[0] / 2)
            center_left = bounds.left + (res[0] / 2)
            center_top = bounds.top - (res[1] / 2)
            center_bottom = bounds.bottom + (res[1] / 2)

            lon = np.arange(center_left, center_right + np.abs(res[0] / 2), np.abs(res[0]), float)
            lat = np.flip(np.arange(center_bottom, center_top + np.abs(res[0] / 2), np.abs(res[1]), float), axis=0)
            lons, lats = np.meshgrid(lon, lat)

            if center_bottom > center_top:
                center_bottom_tmp = center_top
                center_top_tmp = center_bottom
                center_bottom = center_bottom_tmp
                center_top = center_top_tmp
                values = np.flipud(values)
                lats = np.flipud(lats)

            # # Debug
            # plt.figure()
            # plt.imshow(lats)
            # plt.colorbar()
            #
            # # Debug
            # plt.figure()
            # plt.imshow(values)
            # plt.colorbar()
            # plt.show()

            min_lon_round = round(np.min(lons), decimal_round)
            max_lon_round = round(np.max(lons), decimal_round)
            min_lat_round = round(np.min(lats), decimal_round)
            max_lat_round = round(np.max(lats), decimal_round)

            center_right_round = round(center_right, decimal_round)
            center_left_round = round(center_left, decimal_round)
            center_bottom_round = round(center_bottom, decimal_round)
            center_top_round = round(center_top, decimal_round)

            assert min_lon_round == center_left_round
            assert max_lon_round == center_right_round
            assert min_lat_round == center_bottom_round
            assert max_lat_round == center_top_round

            dims = values.shape
            high = dims[0] # nrows
            wide = dims[1] # cols

            bounding_box = [min_lon_round, max_lat_round, max_lon_round, min_lat_round]

            da = create_darray_2d(values, lons, lats, coord_name_x=coord_name_x, coord_name_y=coord_name_y,
                                  dim_name_x=dim_name_x, dim_name_y=dim_name_y, name=var_name)

        else:
            logging.error(' ===> Geographical file ' + file_name + ' format unknown')
            raise NotImplementedError('File type reader not implemented yet')
    else:
        logging.error(' ===> Geographical file ' + file_name + ' not found')
        raise IOError('Geographical file location or name is wrong')

    return da, wide, high, proj, transform, bounding_box, no_data, dim_name_x, dim_name_y
# # -------------------------------------------------------------------------------------
