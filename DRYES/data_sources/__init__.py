import datetime
import xarray as xr

from typing import Optional
from abc import ABC, abstractmethod

from ..data_processes import Grid

class DRYESDataSource(ABC):

    @abstractmethod
    def get_data(self, space: Grid, time: Optional[datetime.datetime] = None) -> xr.Dataset:
        """
        Get data from the data source as an xarray.Dataset.
        for a single time. This is a mandatory method for all subclasses.
        """

    @abstractmethod
    def get_times(time_start: datetime.datetime, time_end: datetime.datetime) -> datetime.datetime:
        """
        Get a list of times between two dates.
        This is a mandatory method for all subclasses.
        """

    @abstractmethod
    def get_timerange(self) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Get the timerange of validity of the data source.
        This is a mandatory method for all subclasses.
        """