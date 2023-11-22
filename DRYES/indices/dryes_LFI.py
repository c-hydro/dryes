from datetime import datetime, timedelta
import numpy as np
import os

from typing import List

from .dryes_index import DRYESIndex
from ..variables.dryes_variable import DRYESVariable
from ..time_aggregation import TimeAggregation

from ..lib.time import TimeRange, doy_to_md
from ..lib.parse import substitute_values
from ..lib.io import get_data, check_data, save_dataarray_to_geotiff
from ..lib.log import log, setup_logging

class DRYESLFI(DRYESIndex):
    def __init__(self, input_variable: DRYESVariable,
                 timesteps_per_year: int,
                 options: dict,
                 output_paths: dict,
                 log_file: str = 'DRYES_log.txt') -> None:

        setup_logging(log_file)
        self.input_variable = input_variable
        self.timesteps_per_year = timesteps_per_year
        options = self.check_options(options)

        self.options = options
        self.output_paths = substitute_values(output_paths, output_paths, rec = False)
        self.get_cases()

    def check_options(self, options: dict) -> dict:
        # check if the options are correct
        if not 'threshold' in options:
            log('No threshold specified, using default threshold of 0.05.')
            options['threshold'] = 0.05
        if not 'thr_window' in options:
            log('No threshold window specified, using default window of 31 days.')
            options['thr_window'] = 31
        
        opts = {}
        opts['threshold'] = options['threshold']
        opts['thr_window'] = options['thr_window']

        return opts

    def compute(self, current: TimeRange, reference: TimeRange) -> None:
        # for the LFI, this needs to be separate from the compute method in the parent class
        # because the LFI does a lot of operations before the time aggregation, like computing the thresholds
        # the time aggregation, in the LFI is only used to determine the timesteps of the final output

        # get the range for the data that is needed
        data_range = TimeRange(reference.start, current.end)

        # get the data -> this will both gather and compute the data (checking if it is already available)
        self.input_variable.make(data_range)

        # calculate thresholds
        self.thresholds = self.calc_thresholds(reference)
        
        # calculate the deficit
        self.deficit = self.calc_deficit(data_range)

        breakpoint()

        pass

    def calc_thresholds(self, reference: TimeRange) -> List[str]:

        log('Starting thresholds calculation')

        dest_parameters = self.output_paths['parameters']
        destination = substitute_values(dest_parameters, {'par': 'thr', 'history_start': reference.start, "history_end": reference.end})
        cases_destinations = [substitute_values(destination, case['tags']) for case in self.cases]

        start_year = reference.start.year
        end_year   = reference.end.year

        input_path = self.input_variable.path

        # loop through all days in a year
        for doy in range(1,366):
            # get the month and day (this ignores leap years)
            month, day = doy_to_md(doy)
            log(f'Calculating thresholds for {day}/{month}...')
            for case in self.cases:
                this_destination = cases_destinations[case['id']]
                # check if this month and day have already been done: we can put a random year here because %Y is not in destination
                if check_data(this_destination, datetime(2000, month, day)): continue

                # get all dates within the window of this month and day in the reference period
                this_thr = case['options']['threshold']
                this_halfwindow = np.floor(case['options']['thr_window']/2)
                these_central_dates = [datetime(year, month, day) for year in range(start_year, end_year + 1)]
                these_dates = []
                for date in these_central_dates:
                    this_date = date - timedelta(days = this_halfwindow)
                    while this_date <= date + timedelta(days = this_halfwindow):
                        if this_date >= reference.start and this_date <= reference.end:
                            these_dates.append(this_date)
                        this_date += timedelta(days = 1)

                # if there are no dates, skip
                if len(these_dates) == 0: continue
                # get the data for these dates
                these_data = [get_data(input_path, date) for date in these_dates]
                data_template = these_data[0].copy()
                these_data = np.stack(these_data, axis = 0)
                # calculate the threshold
                threshold_data = np.quantile(these_data, this_thr, axis = 0)
                # save the threshold
                threshold = data_template.copy(data = threshold_data)
                save_dataarray_to_geotiff(threshold, datetime(2000, month, day).strftime(this_destination))

        dest_dir = os.path.dirname(destination)
        log(f'Thresholds calculated and saved to {dest_dir}.')

        return cases_destinations

def calc_deficit_singlepixel(data: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    # calculate the deficit for a single pixel
    deficit = np.zeros(data.shape[0])
    deficit[data < threshold] = threshold[data < threshold] - data[data < threshold]
    return deficit

def pool_deficit_singlepixel(deficit: np.ndarray) -> np.ndarray:
    # pool the deficit for a single pixel
    pooled_deficit = np.zeros(deficit.shape[0])
    for i in range(deficit.shape[0]):
        pooled_deficit[i] = np.sum(deficit[i:])
    return pooled_deficit