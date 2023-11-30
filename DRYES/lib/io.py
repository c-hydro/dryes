import xarray as xr
import rioxarray
import os
import numpy as np
from functools import lru_cache

from datetime import datetime, timedelta
from typing import Optional, Iterator, Callable

from .time import TimeRange

def save_dataset_to_geotiff(data: xr.Dataset, output_path: str, addn: str = '') -> int:
    if addn != '' and addn[0] != '_': addn = '_' + addn
    for i, var in enumerate(data.data_vars):
        output_file = os.path.join(output_path, f'{var}{addn}.tif')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        this_data = data[var].rio.write_nodata(np.nan, inplace=True)
        this_data.rio.to_raster(output_file, compress = 'lzw')

    return i + 1

def save_dataarray_to_geotiff(data: xr.DataArray, output_file: str,
                              metadata: dict = {}) -> bool:
    # create the directory if it does not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # add metadata
    metadata.update(data.attrs)
    metadata['time_produced'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data.attrs = metadata
    if 'name' in metadata: data.name = metadata['name']

    # save the data to a geotiff
    data = data.rio.write_nodata(np.nan, inplace=True)
    data.rio.to_raster(output_file, compress = 'lzw')

    return True

def check_data(path_pattern, time: Optional[datetime] = None) -> bool:
    path = time.strftime(path_pattern) if time is not None else path_pattern
    return os.path.isfile(path)

@lru_cache(maxsize=120)
def get_data(path_pattern, time: Optional[datetime] = None) -> xr.DataArray:
    path = time.strftime(path_pattern) if time is not None else path_pattern
    if check_data(path):
        data = rioxarray.open_rasterio(path)
        return data
    else:
        return None

def check_data_range(path_pattern, time_range: TimeRange) -> bool|Iterator[datetime]:
    """
    Yields the timesteps in the TimeRange that are available locally
    """
    
    time = time_range.start
    while time <= time_range.end:
        if check_data(path_pattern, time):
            yield time
        time += timedelta(days=1)