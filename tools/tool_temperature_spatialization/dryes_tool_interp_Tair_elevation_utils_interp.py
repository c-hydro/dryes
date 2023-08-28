# -------------------------------------------------------------------------------------
# Library
import logging
import numpy as np
import rasterio
import rasterio.crs
import tempfile
import os
import subprocess

import xarray as xr
from sklearn import linear_model
from random import randint
from datetime import datetime
from shutil import rmtree

from dryes_tool_interp_Tair_elevation_utils_generic import create_darray_2d

logging.getLogger('rasterio').setLevel(logging.WARNING)

# -------------------------------------------------------------------------------------
# Method to interpolate based on homogeneous regions and elevation (includes additional QAQC)
def interp_elevation_by_regions(df_dati, df_stations, info_filters, da_domain_target, da_DEM, da_homogeneous_regions):

    # get homogeneous regions and elevation for each station
    lon_query = xr.DataArray(df_stations['lon'], dims="points")
    lat_query = xr.DataArray(df_stations['lat'], dims="points")
    homogeneous_regions_stations = da_homogeneous_regions.sel(lon=lon_query, lat=lat_query, method="nearest") #I checked .sel with QGIS and it yields expected results.
    elevation_stations = da_DEM.sel(lon=lon_query, lat=lat_query, method="nearest")
    logging.info(' ---> Mean elevation of stations: ' + str(np.round(np.nanmean(elevation_stations), 2)) + ' m asl')

    # get list of homogeneous regions
    list_regions = np.unique(np.ravel(da_homogeneous_regions.values))
    list_regions = list_regions[list_regions > 0]
    logging.info(' ---> Total of sampled homogeneous regions: ' + str(list_regions.shape[0]))

    # loop on this list and do the work for each region; if data points are insufficient, skip computations!
    temp_map_target = np.empty([da_domain_target.shape[0], da_domain_target.shape[1]])*np.nan
    for i, region_id in enumerate(list_regions):

        # determine available stations in this region
        df_stations_this_region = df_stations[homogeneous_regions_stations.values == region_id]

        if df_stations_this_region.shape[0] >= info_filters['minimum_number_sensors_in_region']:

            logging.info(' ---> Homogeneous region: ' + str(region_id) + ', number of stations: ' + str(df_stations_this_region.shape[0]))
            logging.info(' ---> Regression will be computed!')

            df_dati_this_region = np.squeeze(df_dati.loc[:, homogeneous_regions_stations.values == region_id].values)
            elevations_this_region = elevation_stations.values[homogeneous_regions_stations.values == region_id]

            # we drop NaN based on both x and y
            mask = ~np.isnan(elevations_this_region)
            elevations_this_region = elevations_this_region[mask]
            df_dati_this_region = df_dati_this_region[mask]
            mask2 = ~np.isnan(df_dati_this_region)
            elevations_this_region = elevations_this_region[mask2]
            df_dati_this_region = df_dati_this_region[mask2]

            # compute linear regression
            elevations_this_region = elevations_this_region.reshape((-1, 1))  #this is needed to use .fit in LinearReg
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(elevations_this_region, df_dati_this_region)
            r2 = regr.score(elevations_this_region, df_dati_this_region)
            logging.info(' ---> r2 first guess: ' + str(np.round(r2, 3)))

            #compute residuals, remove outliers, re-run regression
            residuals = abs(df_dati_this_region - regr.predict(elevations_this_region))
            df_dati_this_region_filtered = df_dati_this_region[residuals < info_filters['threshold_elevation']]
            elevations_this_region_filtered = elevations_this_region[residuals < info_filters['threshold_elevation']]
            regr2 = linear_model.LinearRegression(fit_intercept=True)
            regr2.fit(elevations_this_region_filtered, df_dati_this_region_filtered)
            r2_2 = regr2.score(elevations_this_region_filtered, df_dati_this_region_filtered)
            logging.info(' ---> Removed ' + str( np.shape(elevations_this_region)[0] -
                                                 np.shape(elevations_this_region_filtered)[0]) + ' stations due to residuals!')
            logging.info(' ---> r2 second guess: ' + str(np.round(r2_2, 3)))

            if r2_2 >= info_filters['minimum_r2']:
                #if r2 is higher than threshold, then apply. Otherwise, do nothing!
                #note that we use here r2_2, so the output of the second linear regression...
                temp_map_target[da_homogeneous_regions.values == region_id] = \
                    regr2.coef_[0]*da_DEM.values[da_homogeneous_regions.values == region_id] + regr2.intercept_
                logging.info(' ---> r2 second guess was higher than threshold, data spatialized!')

                # plt.figure()
                # plt.imshow(temp_map_target)
                # plt.savefig('temp_map_target.png')
                # plt.close()

            else:
                logging.warning(' ---> r2 second guess was LOWER than threshold, data NOT spatialized!')

        else:
            logging.info(' ---> Homogeneous region: ' + str(region_id) + ', number of stations: ' + str(
                df_stations_this_region.shape[0]))
            logging.warning(' ---> Regression will NOT be computed!')

    # plt.figure()
    # plt.imshow(temp_map_target)
    # plt.savefig('temp_map_target_final.png')
    # plt.close()

    # now we must fill NaNs in maps using a national lapse rate
    elevations_all = elevation_stations.values
    df_dati_all = np.squeeze(df_dati.values)

    # we drop NaN based on both x and y
    mask = ~np.isnan(elevations_all)
    elevations_all = elevations_all[mask]
    df_dati_all = df_dati_all[mask]
    mask2 = ~np.isnan(df_dati_all)
    elevations_all = elevations_all[mask2]
    df_dati_all = df_dati_all[mask2]

    #we compute regression
    elevations_all = elevations_all.reshape((-1, 1)) #this is needed to use .fit in LinearReg
    regr_national = linear_model.LinearRegression(fit_intercept=True)
    regr_national.fit(elevations_all, df_dati_all)
    r2_national = regr_national.score(elevations_all, df_dati_all)
    logging.info(' ---> r2 national: ' + str(np.round(r2_national, 3)))

    #we apply in nan areas
    temp_map_target[np.isnan(temp_map_target)] = \
        regr_national.coef_[0] * da_DEM.values[np.isnan(temp_map_target)] + regr_national.intercept_

    # plt.figure()
    # plt.imshow(temp_map_target)
    # plt.colorbar()
    # plt.savefig('temp_map_target_final_filled.png')
    # plt.close()

    #return as Data Array
    lons, lats = np.meshgrid(da_domain_target.lon, da_domain_target.lat)
    map_t_da = create_darray_2d(temp_map_target, lons, lats, name='temp',
                                coord_name_x='lon', coord_name_y='lat',
                                dim_name_x='lon', dim_name_y='lat')
    return map_t_da

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to perform 2d interpolation of residuals (from fp-hyde)
def interp_point2grid(data_in_1d, geox_in_1d, geoy_in_1d, geox_out_2d, geoy_out_2d, epsg_code='4326',
                      interp_no_data=-9999.0, interp_radius_x=None, interp_radius_y=None,
                      interp_method='nearest', interp_option=None,
                      folder_tmp=None, var_name_data='values', var_name_geox='x', var_name_geoy='y',
                      n_cpu=1):

    # Define layer name (using a random string)
    var_name_layer = random_string()

    # Define temporary folder
    if folder_tmp is None:
        folder_tmp = tempfile.mkdtemp()

    if not os.path.exists(folder_tmp):
        make_folder(folder_tmp)

    # Check interpolation radius x and y
    if (interp_radius_x is None) or (interp_radius_y is None):
        logging.error(' ===> Interpolation radius x or y are undefined.')
        raise ValueError('Radius must be defined')

    # Define temporary file(s)
    file_name_csv = os.path.join(folder_tmp, var_name_layer + '.csv')
    file_name_vrt = os.path.join(folder_tmp, var_name_layer + '.vrt')
    file_name_tiff = os.path.join(folder_tmp, var_name_layer + '.tiff')

    # Define geographical information
    geox_out_min = np.min(geox_out_2d)
    geox_out_max = np.max(geox_out_2d)
    geoy_out_min = np.min(geoy_out_2d)
    geoy_out_max = np.max(geoy_out_2d)
    geo_out_cols = geox_out_2d.shape[0]
    geo_out_rows = geoy_out_2d.shape[1]

    # Define dataset for interpolating function
    data_in_ws = np.zeros(shape=[data_in_1d.shape[0], 3])
    data_in_ws[:, 0] = geox_in_1d
    data_in_ws[:, 1] = geoy_in_1d
    data_in_ws[:, 2] = data_in_1d

    # Create csv file
    create_point_csv(file_name_csv, data_in_ws, var_name_data, var_name_geox, var_name_geoy)

    # Create vrt file
    create_point_vrt(file_name_vrt, file_name_csv, var_name_layer, var_name_data, var_name_geox, var_name_geoy)

    # Grid option(s)
    if interp_method == 'nearest':
        if interp_option is None:
            interp_option = ('-a nearest:radius1=' + str(interp_radius_x) + ':radius2=' +
                             str(interp_radius_y) + ':angle=0.0:nodata=' + str(interp_no_data))
    elif interp_method == 'idw':
        if interp_option is None:
            interp_option = ('-a invdist:power=2.0:smoothing=0.0:radius1=' + str(interp_radius_x) + ':radius2=' +
                             str(interp_radius_y) + ':angle=0.0:nodata=' + str(interp_no_data))
    else:
        interp_option = None

    # Execute line command definition (using gdal_grid)
    line_command = ('gdal_grid -zfield "' + var_name_data + '"  -txe ' +
                    str(geox_out_min) + ' ' + str(geox_out_max) + ' -tye ' +
                    str(geoy_out_min) + ' ' + str(geoy_out_max) + ' -a_srs EPSG:' + epsg_code + ' ' +
                    interp_option + ' -outsize ' + str(geo_out_rows) + ' ' + str(geo_out_cols) +
                    ' -of GTiff -ot Float32 -l ' + var_name_layer + ' ' +
                    file_name_vrt + ' ' + file_name_tiff + ' --config GDAL_NUM_THREADS ' + str(n_cpu))

    # Execute algorithm
    [std_out, std_err, std_exit] = exec_process(command_line=line_command)

    # Read data in tiff format and get values
    data_out_obj = rasterio.open(file_name_tiff)
    data_out_3d = data_out_obj.read()

    # Image postprocessing to obtain 2d, south-north, east-west data
    data_out_2d = data_out_3d[0, :, :]
    data_out_2d = np.flipud(data_out_2d)

    # Delete tmp file(s)
    if os.path.exists(file_name_csv):
        os.remove(file_name_csv)
    if os.path.exists(file_name_vrt):
        os.remove(file_name_vrt)
    if os.path.exists(file_name_tiff):
        os.remove(file_name_tiff)

    #set missing to nan
    data_out_2d[data_out_2d == interp_no_data] = np.nan

    return data_out_2d

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to create a random string
def random_string(string_root='temporary', string_separetor='_', rand_min=0, rand_max=1000):

    # Rand number
    rand_n = str(randint(rand_min, rand_max))
    # Rand time
    rand_time = datetime.now().strftime('%Y%m%d-%H%M%S_%f')
    # Rand string
    rand_string = string_separetor.join([string_root, rand_time, rand_n])

    return rand_string
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to make folder
def make_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to create csv ancillary file
def create_point_csv(file_name_csv, var_data, var_name_data='values', var_name_geox='x', var_name_geoy='y',
                     file_format='%10.4f', file_delimiter=','):

    with open(file_name_csv, 'w') as file_handle:
        file_handle.write(var_name_geox + ',' + var_name_geoy + ',' + var_name_data + '\n')
        np.savetxt(file_handle, var_data, fmt=file_format, delimiter=file_delimiter, newline='\n')
        file_handle.close()
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to create vrt ancillary file
def create_point_vrt(file_name_vrt, file_name_csv, var_name_layer,
                     var_name_data='values', var_name_geox='x', var_name_geoy='y'):

    with open(file_name_vrt, 'w') as file_handle:
        file_handle.write('<OGRVRTDataSource>\n')
        file_handle.write('    <OGRVRTLayer name="' + var_name_layer + '">\n')
        file_handle.write('        <SrcDataSource>' + file_name_csv + '</SrcDataSource>\n')
        file_handle.write('    <GeometryType>wkbPoint</GeometryType>\n')
        file_handle.write('    <LayerSRS>WGS84</LayerSRS>\n')
        file_handle.write(
            '    <GeometryField encoding="PointFromColumns" x="' +
            var_name_geox + '" y="' + var_name_geoy + '" z="' + var_name_data + '"/>\n')
        file_handle.write('    </OGRVRTLayer>\n')
        file_handle.write('</OGRVRTDataSource>\n')
        file_handle.close()

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to execute process
def exec_process(command_line=None, command_path=None):

    try:

        # Info command-line start
        logging.info(' ---> Process execution: ' + command_line + ' ... ')

        # Execute command-line
        if command_path is not None:
            os.chdir(command_path)
        process_handle = subprocess.Popen(
            command_line, shell=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Read standard output
        while True:
            string_out = process_handle.stdout.readline()
            if isinstance(string_out, bytes):
                string_out = string_out.decode('UTF-8')

            if string_out == '' and process_handle.poll() is not None:

                if process_handle.poll() == 0:
                    break
                else:
                    logging.error(' ===> Run failed! Check command-line settings!')
                    raise RuntimeError('Error in executing process')
            if string_out:
                logging.info(str(string_out.strip()))

        # Collect stdout and stderr and exitcode
        std_out, std_err = process_handle.communicate()
        std_exit = process_handle.poll()

        if std_out == b'' or std_out == '':
            std_out = None
        if std_err == b'' or std_err == '':
            std_err = None

        # Check stream process
        stream_process(std_out, std_err)

        # Info command-line end
        logging.info(' ---> Process execution: ' + command_line + ' ... DONE')
        return std_out, std_err, std_exit

    except subprocess.CalledProcessError:
        # Exit code for process error
        logging.error(' ===> Process execution FAILED! Errors in the called executable!')
        raise RuntimeError('Errors in the called executable')

    except OSError:
        # Exit code for os error
        logging.error(' ===> Process execution FAILED!')
        raise RuntimeError('Executable not found!')

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to stream process
def stream_process(std_out=None, std_err=None):

    if std_out is None and std_err is None:
        return True
    else:
        logging.warning(' ===> Exception occurred during process execution!')
        return False
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to delete folder (and check if folder exists)
def delete_folder(path_folder):
    # Check folder status
    if os.path.exists(path_folder):
        # Remove folder (file only-read too)
        rmtree(path_folder, ignore_errors=True)

# -------------------------------------------------------------------------------------