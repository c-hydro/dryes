from abc import ABC, abstractmethod
from typing import Optional

from ..lib.time import TimeRange
from ..lib.space import Grid

class DRYESVariable(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def make(self, grid: Grid, time_range: Optional[TimeRange] = None) -> None:
        pass