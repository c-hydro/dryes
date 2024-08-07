from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from deprecated import deprecated

from typing import Iterable

@deprecated(reason="Use dryes.tools.timestepping.TimeRange instead")
class TimeRange():
    def __init__(self, start: datetime, end: datetime):
        """
        Creates a TimeRange object. Useful to download data from a data source.
        If a timerange object is passed to data source, the data source will
        download all the data between the two dates.
        """
        self.isrange = True
        self.start = start
        self.end = end
    
    def __iter__(self):
        self.now = self.start
        return self
    
    def __next__(self):
        if self.now > self.end:
            raise StopIteration
        now = self.now
        self.now+= timedelta(days=1)
        return now

@deprecated(reason="Use the dryes.tools.timestepping.TimeRange method get_timesteps_from_tsnumber() instead")
def create_timesteps(time_start: datetime, time_end: datetime, n_intervals: int) -> list[datetime]:
    """
    Creates a list of timesteps between two dates.
    n_intervals is the number of subdivisions of the year to consider.
    n_intervals can only take as value 1, 2, 3, 4, 6, 12, 24, 36, 365.

    """
    start_year = time_start.year
    end_year = time_end.year

    if n_intervals not in [1, 2, 3, 4, 6, 12, 24, 36, 365]:
        raise ValueError("Invalid number of intervals. Must be a positive integer that divides 12, 24, or 36 evenly.")

    if n_intervals == 1:
        timesteps =  [datetime(year,     1,   1) for year in range(start_year, end_year + 2)]
    elif n_intervals == 2:
        timesteps =  [datetime(year, month,   1) for year in range(start_year, end_year + 2) for month in [1, 7]]
    elif n_intervals == 3:
        timesteps =  [datetime(year, month,   1) for year in range(start_year, end_year + 2) for month in [1, 5, 9]]
    elif n_intervals == 4:
        timesteps =  [datetime(year, month,   1) for year in range(start_year, end_year + 2) for month in [1, 4, 7, 10]]
    elif n_intervals == 6:
        timesteps =  [datetime(year, month,   1) for year in range(start_year, end_year + 2) for month in [1, 3, 5, 7, 9, 11]]
    elif n_intervals == 12:
        timesteps =  [datetime(year, month,   1) for year in range(start_year, end_year + 2) for month in range(1, 13)]
    elif n_intervals == 24:
        timesteps =  [datetime(year, month, day) for year in range(start_year, end_year + 2) for month in range(1, 13) for day in [1, 16]]
    elif n_intervals == 36:
        timesteps =  [datetime(year, month, day) for year in range(start_year, end_year + 2) for month in range(1, 13) for day in [1, 11, 21]]
    elif n_intervals == 365:
        timesteps =  [time_start]
        new_timestep = time_start
        while True:
            new_timestep = new_timestep + timedelta(days=1)
            if (new_timestep.month, new_timestep.day) != (2, 29):
                timesteps.append(new_timestep)
            if new_timestep > time_end:
                break

    timesteps = [time - timedelta(days=1) for time in timesteps]
    timesteps = [time for time in timesteps if time >= time_start and time <= time_end]

    return timesteps

@deprecated(reason="Use the methods of the dryes.tools.timestepping.TimeRange class instead")
def get_interval(date: datetime, num_intervals: int = 12) -> int:
    if num_intervals not in [1, 2, 3, 4, 6, 12, 24, 36]:
        raise ValueError("Invalid number of intervals. Must be a positive integer that divides 12, 24, or 36 evenly.")

    if num_intervals in [1, 2, 3, 4, 6, 12]:
        interval_length = 12 // num_intervals #this is in months
    elif num_intervals == 24:
        interval_length = 15 #this is in days
    else:
        interval_length = 10 #this is in days
    
    month = date.month
    day = date.day
    
    if num_intervals in [1, 2, 3, 4, 6, 12]:
        interval = np.ceil(month / interval_length)
    else:
        intervals_per_month = num_intervals // 12
        interval = (month - 1) * intervals_per_month + min(np.ceil(day / interval_length), intervals_per_month)
    
    return int(interval)

@deprecated(reason="Use the methods of the dryes.tools.timestepping.TimeRange class instead")
def get_interval_date(date: datetime, num_intervals: int = 12, end: bool = False) -> datetime:
    interval = get_interval(date, num_intervals)
    return(get_date_from_interval(interval, date.year, num_intervals, end))

@deprecated(reason="Use the methods of the dryes.tools.timestepping.TimeRange class instead")
def get_date_from_interval(interval: int, year: int, num_intervals: int = 12, end: bool = False) -> datetime:
    if num_intervals not in [1, 2, 3, 4, 6, 12, 24, 36]:
        raise ValueError("Invalid number of intervals. Must be a positive integer that divides 12, 24, or 36 evenly.")

    if num_intervals in [1, 2, 3, 4, 6, 12]:
        interval_length = 12 // num_intervals #this is in months
    elif num_intervals == 24:
        interval_length = 15 #this is in days
    else:
        interval_length = 10 #this is in days
    
    if num_intervals in [1, 2, 3, 4, 6, 12]:
        day = 1
        month = int((interval - 1) * interval_length + 1)
        if end:
            month += interval_length
            if month > 12:
                month = 1
                year += 1
        date = datetime(year, month, day)
        if end:
            date = date - timedelta(days=1)
        return date
    
    else:
        intervals_per_month = num_intervals // 12
        month = int(np.floor((interval - 1) / intervals_per_month) + 1)
        in_month_interval = (interval - 1) % intervals_per_month + 1
        day = (in_month_interval - 1) * interval_length + 1
        if end:
            if in_month_interval == intervals_per_month:
                month += 1
                day = 1
                if month > 12:
                    month = 1
                    year += 1
                date = datetime(year, month, day) - timedelta(days=1)
            else:
                day += interval_length - 1
                date = datetime(year, month, day)
        else:
            date = datetime(year, month, day)
        return date

@deprecated(reason="Use the methods of the dryes.tools.timestepping.TimeRange class instead")
def doy_to_md(doy:int) -> tuple[int, int]:
    """
    Converts day of year to month and day. Ignores 29 February.
    """
    
    date = datetime(1987, 1, 1) + timedelta(days = doy - 1)
    return (date.month, date.day)

@deprecated(reason="Use the methods of the dryes.tools.timestepping.TimeRange class instead")
def ntimesteps_to_md(timesteps_per_year: int) -> list[tuple[int, int]]:
    timesteps = create_timesteps(datetime(1987, 1, 1), datetime(1987, 12, 31), timesteps_per_year)
    return [(time.month, time.day) for time in timesteps] 

@deprecated(reason="Use dryes.tools.timestepping.get_window instead")
def get_window(time: datetime, size: int, unit: str, start = False) -> TimeRange:
        if unit[-1] != 's': unit += 's'
        if unit in ['months', 'years', 'days', 'weeks']:
            if start:
                time_start = time
                time_end = time + timedelta(days=1) + relativedelta(**{unit: size}) - timedelta(days=2)
            else:
                time_start = time + timedelta(days=1) - relativedelta(**{unit: size})
                time_end = time
        elif unit == 'dekads':
            if start:
                if time.day not in  [1, 11, 21]:
                    raise ValueError('Dekads can only start on the 1st, 11th, or 21st of the month')
                time_start = time
                dekad_start = get_interval(time, 36)
                dekad_end = dekad_start + size - 1
                year = time.year
                while dekad_end > 36:
                    dekad_end -= 36
                    year += 1
                time_end = get_date_from_interval(dekad_end, year, 36, end = True)
            else:
                if (time + timedelta(days=1)).day not in  [1, 11, 21]:
                    raise ValueError('Dekads can only start on the 1st, 11th, or 21st of the month')
                dekad_end   = get_interval(time, 36)
                dekad_start = dekad_end - size + 1
                year = time.year
                while dekad_start < 1:
                    dekad_start += 36
                    year -= 1
                time_start = get_date_from_interval(dekad_start, year, 36)
                time_end = time
        else:
            raise ValueError('Unit for aggregator not recognized: must be one of dekads, months, years, days, weeks')
        return TimeRange(time_start, time_end)

@deprecated(reason="Use dryes.tools.timestepping.Year and its is_leap() method instead")
def is_leap(year: int) -> bool:
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    elif year % 400 != 0:
        return False
    else:
        return True

@deprecated(reason="Use dryes.tools.timestepping.get_date_from_str() instead")
def get_md_dates(years: Iterable[int], month: int, day: int) -> list[datetime]:
    if month == 2 and day in [28, 29]:
        leaps = [year for year in years if is_leap(year)]
        nonleaps = [year for year in years if not is_leap(year)]
        return [datetime(year, 2, 29) for year in leaps] + [datetime(year, 2, 28) for year in nonleaps]
    else:
        return [datetime(year, month, day) for year in years]