import numpy as np
import xarray as xr
import rasterio
from typing import Optional

# grid object useful for regridding datasets
class Grid:
    def __init__(self, source: str) -> None:
        """
        Creates a grid object from a source tif file.
        """
        self.source = source
        self._load_grid()
    
    def _load_grid(self):
        """
        Loads the grid from the source tif file.
        """
        with rasterio.open(self.source) as src:
            self_data = src.read(1)
            invmask = np.logical_or(self_data == 0, np.isnan(self_data), self_data == src.nodata)
            self.mask = np.logical_not(invmask)
            self.shape = self.mask.shape
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            self.resolution = src.res
        
    def apply(self, data: xr.Dataset|xr.DataArray, coord_names:Optional[list[str]] = None, mask:bool = True) -> xr.Dataset|xr.DataArray:
        """
        Applies the grid to a xarray.DataArray.
        """
        if coord_names is None:
            coord_names = get_coord_names(data)

        latname = coord_names[0]
        lonname = coord_names[1]

        transform = self.transform
        shape = self.shape

        # Create a new xarray Dataset that represents the target grid
        lon = np.arange(transform[2], transform[2] + transform[0] * shape[1], transform[0])
        lat = np.arange(transform[5], transform[5] + transform[4] * shape[0], transform[4])
        target = {latname: (lat), lonname: (lon)}        

        # Interpolate the original data onto the target grid
        regridded = data.interp(target, method='nearest')

        # Add metadata
        regridded.rio.write_transform(transform, inplace=True)
        regridded.rio.write_crs(self.crs, inplace=True)
        # Mask the data if requested
        if mask:
            # Reshape regridded to match the target grid, the order of the dimensions
            # needs to be other dimentions, latname, lonname
            regridded = regridded.where(self.mask)

        return regridded

def regrid_from_tif(data: xr.Dataset, grid_file: str, coord_names:Optional[list[str]] = None, mask:bool = False) -> xr.Dataset:
    """
    Regrid a xarray Dataset to a target grid defined by a .tif file.
    """

    if coord_names is None:
        coord_names = get_coord_names(data)

    # Load the .tif file
    with rasterio.open(grid_file) as src:
        out_transform = src.transform
        out_crs = src.crs
        out_shape = (src.height, src.width)
        src_mask = np.logical_or(src.read(1) != 0 , src.read(1) != np.nan)

    latname = coord_names[0]
    lonname = coord_names[1]

    # Create a new xarray Dataset that represents the target grid
    lon = np.arange(out_transform[2], out_transform[2] + out_transform[0] * out_shape[1], out_transform[0])
    lat = np.arange(out_transform[5], out_transform[5] + out_transform[4] * out_shape[0], out_transform[4])
    target = {latname: (lat), lonname: (lon)}

    # Interpolate the original data onto the target grid
    regridded = data.interp(target, method='nearest')

    # Add metadata
    regridded.rio.write_transform(out_transform, inplace=True)
    regridded.rio.write_crs(out_crs, inplace=True)

    # Mask the data if requested
    if mask:
        regridded = regridded.where(src_mask)

    return regridded

def get_coord_names(dataset: xr.Dataset) -> list[str]:
    """
    Searches for the names of the latitude and longitude coordinates in a xarray Dataset.
    """
    coord_names = ['lat', 'lon']
    for coord in dataset.coords:
        if 'lat' in coord.lower():
            coord_names[0] = coord
        elif 'lon' in coord.lower():
            coord_names[1] = coord
    
    return coord_names