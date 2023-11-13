import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import os

def save_dataset_to_geotiff(data: xr.Dataset, output_path: str, addn: str = '') -> int:
    if addn != '': addn = '_' + addn
    for i, var in enumerate(data.data_vars):
        output_file = os.path.join(output_path, f'{var}{addn}.tif')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data[var].rio.to_raster(output_file, compress = 'lzw')

    return i + 1

def save_dataarray_to_geotiff(data: xr.DataArray, output_file: str) -> bool:
    # create the directory if it does not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # save the data to a geotiff
    data.rio.to_raster(output_file, compress = 'lzw')

    return True

    # # Convert the xarray.DataArray to a numpy.ndarray
    # array = data.data

    # # Define the transformation
    # transform = from_origin(0, 0, 1, 1)  # Replace with your actual values

    # # Define the metadata
    # metadata = {
    #     'driver': 'GTiff',
    #     'height': array.shape[1],
    #     'width': array.shape[2],
    #     'count': array.shape[0],
    #     'dtype': array.dtype,
    #     'crs': '+proj=latlong',  # Replace with your actual CRS
    #     'transform': transform,
    # }

    # # Write the array to a GeoTIFF file
    # with rasterio.open(output_path, 'w', **metadata) as dst:
    #     dst.write(array)

