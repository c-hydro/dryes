from typing import Optional
from datetime import datetime, timedelta

from .dryes_variable import DRYESVariable

from ..lib.time import TimeRange
from ..lib.space import Grid
from ..lib.log import log
from ..lib.io import get_data, check_data, save_dataarray_to_geotiff

class LocalVariable(DRYESVariable):
    def __init__(self, path: str,
                 type: str,
                 varname: Optional[str] = None,
                 destination: Optional[str] = None) -> None:
        self.input_path = path
        self.isstatic = type == 'static'
        self.name = varname if varname is not None else 'variable'
        self.path = destination if destination is not None else self.input_path
        self.start = self.get_start()

    def get_times(self, time_start: datetime, time_end: datetime) -> datetime:
        """
        Get a list of times between two dates.
        """
        time = time_start
        while time <= time_end:
            if check_data(self.input_path, time):
                yield time
            time += timedelta(days=1)

    def get_start(self) -> datetime:
        """
        Get the start of the available data.
        """
        time_start = datetime(1900, 1, 1)
        time_end = datetime.now()
        for time in self.get_times(time_start, time_end):
            return time

    def make(self, grid: Grid, time_range: Optional[TimeRange] = None) -> None:
        """
        Gathers all the data from the remote source in the TimeRange,
        also checks that the data is not available yet before gathering it
        """
        if self.input_path == self.path:
            pass
        
        for timestep in self.get_times(time_range.start, time_range.end):
            output_path = timestep.strftime(self.path)
            if check_data(output_path):
                continue
            data = get_data(self.input_path, timestep)
            gridded_data = grid.apply(data)
            output_path = timestep.strftime(self.path)
            gridded_data.name = self.name
            saved = save_dataarray_to_geotiff(gridded_data, output_path)

            log(f' - {self.name}: saved files to {output_path}')