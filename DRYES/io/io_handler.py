from typing import Optional
from datetime import datetime
import numpy as np
import xarray as xr

from ..lib.time import TimeRange

class IOHandler:
    def __init__(self) -> None:
        pass

    def get_data(self, time: Optional[datetime], **kwargs) -> xr.DataArray:
        """
        Get the data for a given time.
        """
        raise NotImplementedError
    
    def write_data(self, data: xr.DataArray, time: Optional[datetime], **kwargs):
        """
        Write the data for a given time.
        """
        raise NotImplementedError
    
    def get_times(self, time_range: TimeRange, **kwargs) -> list[datetime]:
        """
        Get a list of times between two dates.
        """
        raise NotImplementedError

    @property
    def start(self) -> datetime:
        """
        Get the start of the available data.
        """
        raise NotImplementedError
    
    def check_data(self, time: Optional[datetime] = None, **kwargs) -> bool:
        """
        Check if data is available for a given time.
        """
        raise NotImplementedError

    # def get_data_range(self, time_range: TimeRange, **kwargs) -> Iterator[xr.DataArray]:
    #     """
    #     Get the data for a given time range.
    #     """
    #     if time_range.start < self.start:
    #         return None #TODO: log warining or debug
        
    #     for time in self.get_times(time_range, **kwargs):
    #         yield self.get_data(time, **kwargs)

    def get_template(self, **kwargs):
        start_data = self.get_data(time = self.start, **kwargs)
        template = self.make_template_from_data(start_data)
        self.template = template
        return template

    def set_template(self, template: xr.DataArray):
        self.template = template

    @staticmethod
    def make_template_from_data(data: xr.DataArray) -> xr.DataArray:
        """
        Make a template xarray.DataArray from a given xarray.DataArray.
        """
        # make a copy of the data, but fill it with NaNs
        template = data.copy(data = np.full(data.shape, np.nan))

        # clear all attributes
        template.attrs = {}
        template.encoding = {}

        # make crs and nodata explicit as attributes
        template.attrs = {'crs': data.rio.crs.to_wkt(),
                          '_FillValue': np.nan}

        return template