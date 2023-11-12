import datetime

class TimeSteps():
    def __init__(self, timesteps: list[datetime.datetime]):
        """
        Creates a TimeSteps object. Useful to download data from a data source.
        If a timesteps object is passed to data source, the data source will
        download all the data at the specified times.
        """
        self.isrange = False
        self.timesteps = timesteps

class TimeRange():
    def __init__(self, start: datetime.datetime, end: datetime.datetime):
        """
        Creates a TimeRange object. Useful to download data from a data source.
        If a timerange object is passed to data source, the data source will
        download all the data between the two dates.
        """
        self.isrange = True
        self.start = start
        self.end = end