# -------------------------------------------------------------------------------------
# Library
import logging
import rasterio
import numpy as np
import pandas as pd
import os
import xarray as xr
import rioxarray
from rasterio.transform import Affine
from osgeo import gdal, gdalconst
from dryes_monthly_aggregator_generic import create_darray_3d, fill_tags2string

logging.getLogger('rasterio').setLevel(logging.WARNING)

# --------------------------------------------------------------------------------
def load_monthly_from_geotiff(da_domain_in,period_daily, period_monthly,
                                       folder_in, filename_in, template,
                                       aggregation_method, check_range, range,
                                       check_climatology_max, path_climatology_max_in, threshold_climatology_max,
                                       multidaily_cumulative, number_days_cumulative):

    # Load daily data and compute monthly means at the end of each month
    data_month_values = np.zeros(
        shape=(da_domain_in.shape[0], da_domain_in.shape[1])) * np.nan  # initialize container for monthly cumulative values
    data_month_values_ALL = np.zeros(shape=(da_domain_in.shape[0], da_domain_in.shape[1],
                                            period_monthly.__len__())) * np.nan  # initialize container for monthly cumulative values

    for time_i, time_date in enumerate(period_daily):

        # load data
        if multidaily_cumulative:
            # if multidaily_cumulative is true, we loop FORWARD for number_days_cumulative days in search of the data.
            period_search = pd.date_range(pd.to_datetime(time_date),
                                          pd.to_datetime(time_date) + pd.DateOffset(days=number_days_cumulative - 1),
                                          freq="D")

            for time_i_search, time_date_search in enumerate(period_search):

                logging.info(' --> Searching ' + time_date.strftime("%Y-%m-%d %H:%M") + ' over '+ str(number_days_cumulative) + ' days ...')

                path_data = os.path.join(folder_in, filename_in)
                tag_filled = {'source_gridded_sub_path_time': time_date,
                          'source_gridded_datetime': time_date}
                path_data = fill_tags2string(path_data, template, tag_filled)
                if os.path.isfile(path_data):
                    logging.info(' --> Found cumulative for ' + time_date.strftime("%Y-%m-%d %H:%M") + ' at ' +  path_data)
                    break

                # if we DO NOT FIND any data, we can leave path_data wit the last time_date_search, as later
                # in this script we have another if os.path.isfile(path_data) that conditions all further steps

        else:
            # if multidaily_cumulative is false, we use time_date
            path_data = os.path.join(folder_in, filename_in)
            tag_filled = {'source_gridded_sub_path_time': time_date,
                          'source_gridded_datetime': time_date}
            path_data = fill_tags2string(path_data, template, tag_filled)
            logging.info(' --> Looking for daily data at ' + path_data)

        if os.path.isfile(path_data):
            data_this_day = rioxarray.open_rasterio(path_data)
            data_this_day = np.squeeze(data_this_day)

            if multidaily_cumulative:
                data_this_day = data_this_day/number_days_cumulative
                logging.info(' --> Divided cumulative data by  ' + str(number_days_cumulative))


            #resampling
            if data_this_day.shape != da_domain_in.shape:
                coordinates_target = {
                    data_this_day.dims[0]: da_domain_in[da_domain_in.dims[0]].values,
                    data_this_day.dims[1]: da_domain_in[da_domain_in.dims[1]].values}
                data_this_day = data_this_day.interp(coordinates_target, method='nearest')
                logging.info(' --> Resampling ' + time_date.strftime("%Y-%m-%d %H:%M"))

            data_this_day_values = data_this_day.values

            # check range if needed
            if check_range:
                data_this_day_values[data_this_day_values < range[0]] = range[0]
                data_this_day_values[data_this_day_values > range[1]] = range[1]

            logging.info(' --> ' + time_date.strftime("%Y-%m-%d %H:%M") + ' loaded from ' + path_data)
        else:
            data_this_day_values = np.zeros(shape=(da_domain_in.shape[0], da_domain_in.shape[1])) * np.nan
            logging.warning(' ==> ' + time_date.strftime("%Y-%m-%d %H:%M") + ' not found!')

        # plt.figure()
        # plt.imshow(data_this_day_values)
        # plt.savefig('data_this_day_values.png')
        # plt.close()

        # accumulate monthly values
        data_month_values = np.nansum(np.dstack((data_month_values, data_this_day_values)),
                                      2)  # we add daily data to monthly cumulative ignoring nan

        # if end of month, save statistics and reset monthly cumulative matrix
        if time_date.is_month_end:
            if aggregation_method == 'mean':
                data_month_values = data_month_values / time_date.daysinmonth  # compute avg monthly stat
                logging.info(' --> Avg value computed for ' + time_date.strftime("%Y-%m-%d %H:%M"))
            elif aggregation_method == 'sum':
                data_month_values = data_month_values  # keep cumulative values
                logging.info(' --> Cum. value computed for ' + time_date.strftime("%Y-%m-%d %H:%M"))
            else:
                logging.error(' ===> Aggregation method not supported!')
                raise ValueError(' ===> Aggregation method not supported!')

            # check monthly climatology if needed
            if check_climatology_max:
                # load climatology layer
                tag_filled = {'source_gridded_climatology_sub_path_time': time_date,
                              'source_gridded_climatology_datetime': time_date}
                path_climatology = fill_tags2string(path_climatology_max_in, template, tag_filled)
                data_climatology = rioxarray.open_rasterio(path_climatology)
                data_climatology = np.squeeze(data_climatology)

                # regrid
                coordinates_target = {
                    data_climatology.dims[0]: da_domain_in[da_domain_in.dims[0]].values,
                    data_climatology.dims[1]: da_domain_in[da_domain_in.dims[1]].values}
                data_climatology = data_climatology.interp(coordinates_target, method='nearest')

                # set no data to NaN
                data_climatology.values[data_climatology.values == data_climatology._FillValue] = np.nan

                # plt.figure()
                # plt.imshow(data_climatology.values)
                # plt.colorbar()
                # plt.savefig(time_date.strftime("%m") + 'climatology.png')
                # plt.close()

                # apply threshold
                data_month_values[data_month_values > threshold_climatology_max*data_climatology.values] = np.nan

            data_month_values_ALL[:, :, period_monthly.get_loc(time_date)] = data_month_values
            logging.info(
                ' --> Monthly statistics stored in datacube at row ' + str(period_monthly.get_loc(time_date)))

            # plt.figure()
            # plt.imshow(data_month_values)
            # plt.colorbar()
            # plt.savefig(time_date.strftime("%Y%m%d") + 'cum.png')
            # plt.close()

            data_month_values = np.zeros(shape=(da_domain_in.shape[0], da_domain_in.shape[1])) * np.nan  # reset cumulative container

    data_ALL_da = create_darray_3d(data_month_values_ALL,
                                                period_monthly, da_domain_in[da_domain_in.dims[1]].values,
                                                da_domain_in[da_domain_in.dims[0]].values)
    return data_ALL_da
# --------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------
# Method to write file tiff
def write_file_tiff(file_name, file_data, file_wide, file_high, file_geotrans, file_proj,
                    file_metadata=None, file_format=gdalconst.GDT_Float32):

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
        dset_handle.GetRasterBand(file_id + 1).WriteArray(file_data_step)
        dset_handle.GetRasterBand(file_id + 1).SetMetadata(file_metadata_step)
    del dset_handle
# -------------------------------------------------------------------------------------