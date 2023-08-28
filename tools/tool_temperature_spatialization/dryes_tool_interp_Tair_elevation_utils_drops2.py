# -------------------------------------------------------------------------------------
# Library
import logging
import numpy as np
import pandas as pd
import xarray as xr
from drops2 import sensors
from drops2.utils import DropsException
from time import sleep

from dryes_tool_interp_Tair_elevation_utils_geo import read_file_raster

logging.getLogger('rasterio').setLevel(logging.WARNING)

# -------------------------------------------------------------------------------------
# Method to read drops2 data
def GetDrops2(date_from, date_to, aggregation_seconds, group, sensor_class,
              bounding_box_domain_target, invalid_flags, ntry=10, sec_sleep=5):

    sensors_list = None
    while ntry > 0:

        try:

            ntry -= 1
            # get sensor list
            sensors_list = sensors.get_sensor_list(sensor_class, geo_win=(bounding_box_domain_target[0],
                                                                          bounding_box_domain_target[3],
                                                                          bounding_box_domain_target[2],
                                                                          bounding_box_domain_target[1]), group=group)

            if len(sensors_list.list) == 0:
                logging.warning(' ---> No available station for this time step!')
                df_dati = pd.DataFrame()
                dfStations = pd.DataFrame()
                return df_dati, dfStations

            # create df with station metadata
            dfStations = pd.DataFrame(np.array([(p.name, p.lat, p.lng) for p in sensors_list]),
                                      index=np.array([(p.id) for p in sensors_list]), columns=['name', 'lat', 'lon'])

            # get data
            date_from_str = date_from.strftime("%Y%m%d%H%M")
            date_to_str = date_to.strftime("%Y%m%d%H%M")
            df_dati = sensors.get_sensor_data(sensor_class, sensors_list,
                                              date_from_str, date_to_str, aggr_time=aggregation_seconds,
                                              as_pandas=True)
            break

        except DropsException:

            logging.warning(' ---> Problems with downloading Drops2 data, retrying in ' + str(sec_sleep) + 'seconds')

            if ntry >= 0:
                sleep(sec_sleep)
                logging.warning(
                    ' ---> ... ')
            else:
                logging.error(' ===> Problem with extraction from drops2!')
                raise

    if sensors_list is None:
        logging.error(' ===> Problem with extraction from drops2!')
        raise ValueError(' ===> Problem with extraction from drops2!')

    number_stat_initial = sensors_list.list.__len__()
    logging.info(' ---> Extracted ' + str(number_stat_initial) + ' sensors')

    # For cautionary reasons, we may have asked drops2 more hours of data than time_date.
    # So here we extract the row we need...
    df_dati = df_dati.loc[df_dati.index == date_to_str]

    # We remove NaNs and invalid points
    logging.info(' ---> Checking for empty or not-valid series')
    for i_invalid, value_invalid in enumerate(invalid_flags):
        df_dati.values[df_dati.values == value_invalid] = np.nan
    df_dati = df_dati.dropna(axis='columns', how='all')
    dfStations = dfStations.loc[list(df_dati.columns)]

    number_stat_end = dfStations.shape[0]
    number_removed = number_stat_initial - number_stat_end
    logging.info(' ---> Removed ' + str(number_removed) + ' stations')
    logging.info(' ---> Number of available stations is ' + str(number_stat_end))

    return df_dati, dfStations
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# QAQC method based on climatology
def QAQC_climatology(df_dati, dfStations, path_climatology, threshold):

    # load climatology & extract at coordinates
    da_climatology, wide_climatology, high_climatology, proj_climatology, \
        transform__climatology, bounding_box__climatology, no_data_climatology, \
        crs__climatology, lons__climatology, lats__climatology = read_file_raster(path_climatology, coord_name_x='lon',
                                                                                  coord_name_y='lat',dim_name_x='lon',
                                                                                  dim_name_y='lat')
    da_climatology.values[da_climatology.values == no_data_climatology] = np.nan

    # plt.figure()
    # plt.imshow(da_climatology.values)
    # plt.savefig('da_climatology.png')
    # plt.close()

    lon_query = xr.DataArray(dfStations['lon'], dims="points")
    lat_query = xr.DataArray(dfStations['lat'], dims="points")
    t_climatology_points = da_climatology.sel(lon=lon_query, lat=lat_query, method="nearest")

    # compute min e max range
    t_climatology_points_min = t_climatology_points - threshold
    t_climatology_points_max = t_climatology_points + threshold

    # apply threshold
    df_dati[df_dati.values < t_climatology_points_min.values] = np.nan
    df_dati[df_dati.values > t_climatology_points_max.values] = np.nan

    number_stations_before = df_dati.shape[0]
    df_dati = df_dati.dropna(axis='columns', how='all')
    dfStations = dfStations.loc[list(df_dati.columns)]
    number_stations_after = df_dati.shape[0]
    logging.info(' ---> Climatological filter removed ' + str(number_stations_before - number_stations_after) + ' stations')

    return df_dati, dfStations



# -------------------------------------------------------------------------------------