import datetime
import xarray as xr

from typing import Optional
from abc import ABC, abstractmethod

from ..lib.space import Grid

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
    def get_start(self) -> datetime.datetime:
        """
        Get the start of the data source.
        This is a mandatory method for all subclasses.
        """