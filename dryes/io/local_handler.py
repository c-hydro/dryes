from typing import Optional, Generator, Callable
from datetime import datetime
from functools import cached_property
from methodtools import lru_cache
import os
import rioxarray
import numpy as np
import xarray as xr

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
        self.name = name if name is not None else os.path.basename(file).split('.')[0]
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
    
    def _get_times(self, time_range: TimeRange, **kwargs) -> Generator[datetime, None, None]:
        for time in time_range:
            if self.check_data(time, **kwargs):
                yield time
            elif hasattr(self, 'parents') and self.parents is not None:
                if all(parent.check_data(time, **kwargs) for parent in self.parents.values()):
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

    #@lru_cache(maxsize=32)
    def get_data(self, time: Optional[datetime] = None, **kwargs):
        if self.check_data(time, **kwargs):
            data = rioxarray.open_rasterio(self.path(time, **kwargs))

            # ensure that the data has descending latitudes
            y_dim = data.rio.y_dim
            if y_dim is None:
                for dim in data.dims:
                    if 'lat' in dim.lower() | 'y' in dim.lower():
                        y_dim = dim
                        break
            if data[y_dim][0] < data[y_dim][-1]:
                data = data.sortby(y_dim, ascending = False)

            # round the coordinates to 1/1000 of the resolution
            for dim in data.dims:
                if len(data[dim]) == 1:
                    continue
                res = data[dim].values[1] - data[dim].values[0]
                # get the position of the first significant digit
                pos = -int(np.floor(np.log10(abs(res))))
                # round the coordinate to 1/1000 of the resolution (3 significant digits more than the resolution)
                data[dim] = np.round(data[dim].values, pos+3)

            # make sure the nodata value is set to np.nan
            if '_FillValue' in data.attrs and not np.isnan(data.attrs['_FillValue']):
                data = data.where(data != data.attrs['_FillValue'])
                data.attrs['_FillValue'] = np.nan

            if not hasattr(self, 'template') or self.template is None:
                self.template = self.make_template_from_data(data)
            return data
        elif hasattr(self, 'parents') and self.parents is not None:
            parent_data = {name: parent.get_data(time, **kwargs) for name, parent in self.parents.items()}
            data = self.fn(**parent_data)
            self.write_data(data, time, **kwargs)
            return data
        else:
            raise ValueError(f'File {self.path(time, **kwargs)} does not exist.')

    def set_parents(self, parents:dict[str:IOHandler], fn:Callable):
        self.parents = parents
        self.fn = fn

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
        
    def write_data(self, data: xr.DataArray,
                   time: Optional[datetime] = None,
                   time_format: str = '%Y-%m-%d',
                   tags = {},
                   **kwargs):
        
        if data is None or data.size == 0:
            output = self.template
        else:
            output = self.template.copy(data = data)

        output_file = self.path(time, **tags)

        # create the directory if it does not exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # add metadata
        metadata = {}
        
        metadata.update(kwargs)
        if hasattr(data, 'attrs'):
            metadata.update(data.attrs)
        
        metadata['time_produced'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if time is not None: metadata['time'] = time.strftime(time_format)

        metadata['name'] = self.name
        if 'long_name' in metadata:
            metadata.pop('long_name')

        output.attrs.update(metadata)
        output.name = self.name

        # save the data to a geotiff
        output.rio.to_raster(output_file, compress = 'lzw')