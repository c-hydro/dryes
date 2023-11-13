from typing import Optional
from datetime import datetime
import xarray as xr

from ..data_sources import DRYESDataSource
from ..lib.log import setup_logging, log

class DRYESDataChain:

    def __init__(self, name: str, log_file: str):
        """
        Creates a DRYESDataChain object from a settings file.
        """
        self.name = name
        self.log_file = log_file
        setup_logging(log_file)


    def set_data_sources(self, dynamic = Optional[dict[str:DRYESDataSource]], static = Optional[dict[str:DRYESDataSource]]):
        """
        Sets the data sources for the data chain.
        Static sources will only need to be downloaded once, when the chain is set up.
        Dynamic sources will be updated every time the chain is run.
        """
        self.dynamic_sources = dynamic
        self.static_sources  = static