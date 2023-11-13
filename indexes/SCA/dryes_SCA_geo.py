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
import pandas as pd
from time import strftime
import xarray as xr
import rioxarray

from dryes_SCA_utils_generic import create_darray_2d, fill_tags2string

logging.getLogger('rasterio').setLevel(logging.WARNING)

# -------------------------------------------------------------------------------------
# Method to compute avg SCA by mountain region
def avg_SCA_by_mountain_region(period, data_settings, list_mountains, da_domain_in, crs_domain_in, transform_domain_in):
    perc_neve_df = pd.DataFrame(index=period, columns=list_mountains.name_mm)
    for time_i, time_date in enumerate(period):

        # we first look for the final tif file. If it exists, we skip resampling
        path_geotiff_resampled = os.path.join(data_settings['data']['outcome']['folder_SCA_resampled'],
                                              data_settings['data']['outcome']['filename_SCA_resampled'])
        tag_filled = {'outcome_sub_path_time': time_date, 'outcome_datetime': time_date,
                      'outcome_sub_path_time_yr': time_date}
        path_geotiff_resampled = \
            fill_tags2string(path_geotiff_resampled, data_settings['algorithm']['template'], tag_filled)

        if os.path.isfile(path_geotiff_resampled) is False:
            filename_in_h10 = os.path.join(data_settings['data']['input']['folder'],
                                           data_settings['data']['input']['filename'])
            tag_filled = {'source_gridded_sub_path_time': (time_date), 'source_datetime': (time_date)}
            filename_in_h10 = \
                fill_tags2string(filename_in_h10, data_settings['algorithm']['template'], tag_filled)

            # if source file exists, we open it, convert to SCA, resample and save
            if os.path.isfile(filename_in_h10):
                SCA_this_day = h10_converter(filename_in_h10, da_domain_in, data_settings, path_geotiff_resampled,
                                             time_date,
                                             crs_domain_in, transform_domain_in)
                logging.info(' --> SCA for ' + time_date.strftime(format='%Y/%m/%d') + ' loaded from SOURCE: '
                             + filename_in_h10)
            else:
                logging.warning(' --> SCA for ' + time_date.strftime(format='%Y/%m/%d')
                                + 'MISSING from SOURCE: ' + filename_in_h10)
                SCA_this_day = np.empty(np.shape(da_domain_in)) * np.nan
        else:
            SCA_this_day = rioxarray.open_rasterio(path_geotiff_resampled)
            SCA_this_day = np.squeeze(SCA_this_day)
            logging.info(' --> SCA for ' + time_date.strftime(format='%Y/%m/%d') +
                         ' loaded from TIFF: ' + path_geotiff_resampled)

        # now we compute SCA for each mountain group
        for index, row in list_mountains.iterrows():
            number_elements = np.count_nonzero(da_domain_in.values == row.mm_code)  # number of potential pixels
            number_snow = np.count_nonzero(
                np.logical_and(SCA_this_day == 1, da_domain_in.values == row.mm_code))  # number of snow pixels
            number_ground = np.count_nonzero(
                np.logical_and(SCA_this_day == 0, da_domain_in.values == row.mm_code))  # number of ground pixels
            number_nodata = np.count_nonzero(
                np.logical_and(SCA_this_day == -1, da_domain_in.values == row.mm_code))  # number of no data

            perc_snow = number_snow / (number_elements)
            perc_no_data = number_nodata / number_elements

            if perc_no_data < data_settings['index_info']['max_no_data_in_mount_region']:
                perc_neve_df.loc[time_date, row.name_mm] = perc_snow
            else:
                perc_neve_df.loc[time_date, row.name_mm] = np.nan

    return perc_neve_df

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to convert source h10 to resampled sonw map
def h10_converter(path_file, da_domain_in, data_settings, path_output_tiff, time_date,
                  crs_domain, transform_domain):

    path_unzip, extension = path_file.split('.gz')

    # unzip
    file_handle_zip = gzip.GzipFile(path_file, "rb")
    file_data_unzip = file_handle_zip.read()
    file_handle_unzip = open(path_unzip, "wb")
    file_handle_unzip.write(file_data_unzip)
    file_handle_zip.close()
    file_handle_unzip.close()

    # load data from file
    hsaf_dset = netCDF4.Dataset(path_unzip, 'r')
    hsaf_values = hsaf_dset.variables[data_settings['data']['input']['layer_data_name']][:]
    hsaf_lon = hsaf_dset.variables[data_settings['data']['input']['layer_data_lon']][:]
    hsaf_lat = hsaf_dset.variables[data_settings['data']['input']['layer_data_lat']][:]

    # we delete unzipped file
    os.remove(path_unzip)

    # we apply scale factor to lat e lon
    hsaf_lon = hsaf_lon / data_settings['data']['input']['lat_lon_scale_factor']
    hsaf_lat = hsaf_lat / data_settings['data']['input']['lat_lon_scale_factor']

    # we remove space and only keep Earth data
    hsaf_values_unf_tmp = np.ravel(hsaf_values)
    hsaf_lon_unf_tmp = np.ravel(hsaf_lon)
    hsaf_lat_unf_tmp = np.ravel(hsaf_lat)  # ravel flattens the Array to a 1d Array
    hsaf_values_unf = hsaf_values_unf_tmp[hsaf_values_unf_tmp != data_settings['data']['input']['no_data_flag_input']]
    hsaf_lon_unf = hsaf_lon_unf_tmp[hsaf_values_unf_tmp != data_settings['data']['input']['no_data_flag_input']]
    hsaf_lat_unf = hsaf_lat_unf_tmp[hsaf_values_unf_tmp != data_settings['data']['input']['no_data_flag_input']]

    # we regrid
    x_mountain_mask, y_mountain_mask = np.meshgrid(da_domain_in.west_east.values, da_domain_in.south_north.values)
    hsaf_values_on_ref_grid = resample_to_grid({'data_resampled': hsaf_values_unf},
                                   hsaf_lon_unf, hsaf_lat_unf, x_mountain_mask, y_mountain_mask,
                                   fill_values=np.nan)
    hsaf_values_on_ref_grid = hsaf_values_on_ref_grid['data_resampled']

    # plt.figure()
    # plt.imshow(hsaf_values_on_ref_grid)
    # plt.savefig('hsaf_values_on_ref_grid.png')
    # plt.close()

    # we convert to snow mask
    hsaf_values_on_ref_grid_SCA = np.empty(np.shape(hsaf_values_on_ref_grid)) - 1
    hsaf_values_on_ref_grid_SCA[hsaf_values_on_ref_grid == data_settings['data']['input']['snow_flag_input']] = 1
    hsaf_values_on_ref_grid_SCA[hsaf_values_on_ref_grid == data_settings['data']['input']['ground_flag_input']] = 0

    # plt.figure()
    # plt.imshow(hsaf_values_on_ref_grid_SCA)
    # plt.savefig('hsaf_values_on_ref_grid_SCA.png')
    # plt.close()

    # save geotiff
    path_output_tiff_dir, path_output_tiff_name = os.path.split(path_output_tiff)
    if os.path.isdir(path_output_tiff_dir) is False:
        os.mkdir(path_output_tiff_dir)
    layer_out = hsaf_values_on_ref_grid_SCA.astype(np.float32)
    with rasterio.open(path_output_tiff, 'w', height=da_domain_in.shape[0],
                       width=da_domain_in.shape[1], count=1, dtype='float32',
                       crs=crs_domain, transform=transform_domain, driver='GTiff',
                       nodata=-9999,
                       compress='lzw') as out:
        out.write(layer_out, 1)

    return layer_out

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

    return da, wide, high, proj, transform, bounding_box, no_data, crs
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
