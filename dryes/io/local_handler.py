from typing import Optional
from datetime import datetime
from functools import cached_property
from methodtools import lru_cache
import os
import rioxarray
import numpy as np

from .io_handler import IOHandler

from ..utils.time import TimeRange
from ..utils.parse import substitute_string

class LocalIOHandler(IOHandler):
    type = 'local'

    def __init__(self,
                 path: str,
                 file: str,
                 name: Optional[str] = None,
                 format: Optional[str] = None) -> None:
        
        self.dir  = path
        self.file = file
        self.path_pattern = os.path.join(self.dir , file)
        self.name = name if name is not None else os.path.basename(file)
        self.format = format if format is not None else file.split('.')[-1]
        if self.format.lower() in ['tif', 'tiff', 'geotiff']:
            self.format = 'GeoTIFF'
        elif self.format.lower() in ['txt']:
            self.format = 'ASCII'
        else:
            raise ValueError(f'Format {self.format} not supported.')
        self.tags = {}

    @cached_property
    def start(self):
        """
        Get the start of the available data.
        """
        time_start = datetime(1900, 1, 1)
        time_end = datetime.now()
        for time in self._get_times(TimeRange(time_start, time_end)):
            return time
    
    def _get_times(self, time_range: TimeRange, **kwargs) -> datetime:
        for time in time_range:
            if self.check_data(time, **kwargs):
                yield time

    def get_times(self, time_range: TimeRange, **kwargs) -> list[datetime]:
        """
        Get a list of times between two dates.
        """
        return list(self._get_times(time_range, **kwargs))
    
    def path(self, time: Optional[datetime] = None, **kwargs):
        raw_path = substitute_string(self.path_pattern, kwargs)
        path = time.strftime(raw_path) if time is not None else raw_path
        return path

    def check_data(self, time: Optional[datetime] = None, **kwargs) -> bool:
        this_path = self.path(time, **kwargs)
        return os.path.exists(this_path)

    @lru_cache(maxsize=256)
    def get_data(self, time: Optional[datetime] = None, **kwargs):
        if self.check_data(time, **kwargs):
            data = rioxarray.open_rasterio(self.path(time, **kwargs))
            if not hasattr(self, 'template') or self.template is None:
                self.template = self.make_template_from_data(data)
            return data
        else:
            raise ValueError(f'File {self.path(time, **kwargs)} does not exist.')

    def update(self, in_place = False, **kwargs):
        if in_place:
            self.dir  = substitute_string(self.dir, kwargs)
            self.file = substitute_string(self.file, kwargs)
            self.path_pattern = self.path(**kwargs)
            self.tags.update(kwargs)
            return self
        else:
            new_path = substitute_string(self.dir, kwargs)
            new_file = substitute_string(self.file, kwargs)
            new_name = self.name
            new_format = self.format
            new_handler = LocalIOHandler(new_path, new_file, new_name, new_format)
            new_handler.template = self.template
            new_tags = self.tags.copy()
            new_tags.update(kwargs)
            new_handler.tags = new_tags
            return new_handler
        
    def write_data(self, data: np.ndarray,
                   time: Optional[datetime] = None,
                   time_format: str = '%Y-%m-%d', **kwargs):
        
        if data is None or data.size == 0:
            output = self.template
        else:
            output = self.template.copy(data = data)

        output_file = self.path(time, **kwargs)

        # create the directory if it does not exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # add metadata
        metadata = {'name': self.name,
                    'time_produced': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        if time is not None: metadata['time'] = time.strftime(time_format)
        metadata.update(self.tags)
        metadata.update(kwargs)
        output.attrs.update(metadata)
        
        output.name = self.name

        # save the data to a geotiff
        output.rio.to_raster(output_file, compress = 'lzw')