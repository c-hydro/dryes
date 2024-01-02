from __future__ import annotations

from typing import Optional

from ..lib.time import TimeRange

from .io_handler import IOHandler

class RemoteIOHandler(IOHandler):
    def __init__(self) -> None:
        raise NotImplementedError

    def get_data(self, time_range: Optional[TimeRange] = None) -> None:
        """
        Gathers all the data from the remote source in the TimeRange,
        also checks that the data is not available yet before gathering it
        """
        raise NotImplementedError