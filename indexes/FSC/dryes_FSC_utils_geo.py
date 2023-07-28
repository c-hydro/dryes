# -------------------------------------------------------------------------------------
# Library
import logging
import numpy as np
import rasterio
import rasterio.crs
from rasterio.enums import Resampling
import os
import matplotlib.pylab as plt
import gzip
import netCDF4
from pytesmo.grid.resample import resample_to_grid

from dryes_FSC_utils_generic import create_darray_2d

logging.getLogger('rasterio').setLevel(logging.WARNING)

# -------------------------------------------------------------------------------------
# Method to convert source h12 to resampled map
def h12_converter(path_file, path_file_tiff, da_domain_out, crs_domain_out, transform_domain_out, path_cdo, layer_data_name, layer_data_lon,
                  layer_data_lat=1, lat_lon_scale_factor=1, valid_range=[0,100]):

    path_unzip, extension = path_file.split('.gz')

    # unzip
    file_handle_zip = gzip.GzipFile(path_file, "rb")
    file_data_unzip = file_handle_zip.read()
    file_handle_unzip = open(path_unzip, "wb")
    file_handle_unzip.write(file_data_unzip)
    file_handle_zip.close()
    file_handle_unzip.close()

    # convert grib to nc
    destfld, destfile = os.path.split(path_file)
    filein = path_unzip
    fileout = os.path.join(destfld, 'tmp_regrid.nc')
    cdocmd = path_cdo + ' -f nc copy'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)

    # load data from file
    hsaf_dset = netCDF4.Dataset(fileout, 'r')
    hsaf_values = hsaf_dset.variables[layer_data_name][:]
    hsaf_lon = hsaf_dset.variables[layer_data_lon][:]
    hsaf_lat = hsaf_dset.variables[layer_data_lat][:]

    # we apply scale factor to lat e lon
    hsaf_lon = hsaf_lon / lat_lon_scale_factor
    hsaf_lat = hsaf_lat / lat_lon_scale_factor

    # we regrid
    x_out, y_out = np.meshgrid(da_domain_out[da_domain_out.dims[1]].values, da_domain_out[da_domain_out.dims[0]].values)
    x_in, y_in = np.meshgrid(hsaf_lon, hsaf_lat)
    hsaf_values_on_ref_grid = resample_to_grid({'data_resampled': np.ravel(hsaf_values)},
                                   np.ravel(x_in), np.ravel(y_in), x_out, y_out,
                                   fill_values=np.nan)
    hsaf_values_on_ref_grid = hsaf_values_on_ref_grid['data_resampled']

    # we set to NaN outside the valid range
    hsaf_values_on_ref_grid[hsaf_values_on_ref_grid > valid_range[1]] = np.nan
    hsaf_values_on_ref_grid[hsaf_values_on_ref_grid < valid_range[0]] = np.nan

    # plt.figure()
    # plt.imshow(hsaf_values_on_ref_grid)
    # plt.colorbar()
    # plt.savefig('hsaf_values_on_ref_grid.png')
    # plt.close()

    # save to tiff
    dir_file_tiff, name_file_tiff = os.path.split(path_file_tiff)
    if os.path.isdir(dir_file_tiff) is False:
        os.makedirs(dir_file_tiff)
    layer_out = hsaf_values_on_ref_grid.astype(np.float32)
    with rasterio.open(path_file_tiff, 'w', height=da_domain_out.shape[0],
                       width=da_domain_out.shape[1], count=1, dtype='float32',
                       crs=crs_domain_out, transform=transform_domain_out, driver='GTiff',
                       nodata=-9999,
                       compress='lzw') as out:
        out.write(layer_out, 1)

    # we delete all temp files
    os.remove(path_unzip)
    os.remove(fileout)

    # return layer_out
    return hsaf_values_on_ref_grid

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to get a raster ascii file
def read_file_raster(file_name, file_proj='epsg:4326', var_name='land',
                     coord_name_x='west_east', coord_name_y='south_north',
                     dim_name_x='west_east', dim_name_y='south_north', no_data_default=-9999.0, scale_factor=1):

    if os.path.exists(file_name):
        if (file_name.endswith('.txt') or file_name.endswith('.asc')) or file_name.endswith('.tif'):

            with rasterio.open(file_name, mode='r+') as dset:

                # resample data to target
                # source: https://rasterio.readthedocs.io/en/latest/topics/resampling.html
                data = dset.read(
                    out_shape=(
                        dset.count,
                        int(dset.height * scale_factor),
                        int(dset.width * scale_factor)
                    ),
                    resampling=Resampling.mode
                )

                # scale image transform
                transform = dset.transform * dset.transform.scale(
                    (dset.width / data.shape[-1]),
                    (dset.height / data.shape[-2])
                )

                #Get ancillary info
                crs = dset.crs
                proj = dset.crs.wkt
                bounds = rasterio.transform.array_bounds(data.shape[-2], data.shape[-1], transform)
                bounds = rasterio.coords.BoundingBox(bounds[0], bounds[1], bounds[2], bounds[3])
                no_data = dset.nodata
                res = (abs(transform.a), abs(transform.e))  #we take resolution from the delta_x in the transform, assuming delta_x and delta_y are the same
                values = data[0, :, :]

            # Define no data if none or nan
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

    return da, wide, high, proj, transform, bounding_box, no_data, crs, lons, lats
# -------------------------------------------------------------------------------------

# method to resample data
def resample_data(file_dframe, geo_x_out_2d, geo_y_out_2d,
                  var_name_data='surface_soil_moisture', var_name_geo_x='longitude', var_name_geo_y='latitude',
                  coord_name_x='longitude', coord_name_y='latitude', dim_name_x='longitude', dim_name_y='latitude',
                  search_radius_fill_nan=18000):

    data_in = file_dframe[var_name_data].values
    geo_x_in = file_dframe[var_name_geo_x].values
    geo_y_in = file_dframe[var_name_geo_y].values

    data_masked = resample_to_grid({var_name_data: data_in}, geo_x_in, geo_y_in, geo_x_out_2d, geo_y_out_2d,
                                   fill_values=np.nan, search_rad=search_radius_fill_nan)
    data_grid = data_masked[var_name_data].data

    # plt.figure()
    # plt.imshow(data_grid)
    # plt.colorbar()
    # plt.savefig('SSMI_resampled.png')
    # plt.close()

    data_out = create_darray_2d(
        data_grid, geo_x_out_2d[0, :], geo_y_out_2d[:, 0], name=var_name_data,
        coord_name_x=coord_name_x, coord_name_y=coord_name_y,
        dim_name_x=dim_name_x, dim_name_y=dim_name_y)

    return data_out
# -------------------------------------------------------------------------------------