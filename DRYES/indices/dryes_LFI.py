from datetime import datetime, timedelta
import numpy as np
import os

from typing import List

from .dryes_index import DRYESIndex

from ..lib.time import TimeRange, doy_to_md
from ..lib.parse import substitute_values
from ..lib.io import get_data, check_data, save_dataarray_to_geotiff
from ..lib.log import log

class DRYESLFI(DRYESIndex):

    # this flag is used to determine if we need to process all timesteps continuously
    # or if we can have gaps in the processing
    # the LFI always works continuously (for other indices, it depends on the time aggregation)
    iscontinuous = True

    # default options for the LFI
    default_options = {
        'threshold'  : 0.05,
        'thr_window' : 31
    }

    parameters = ('Qthreshold', 'lambda')

    def calc_parameters(self, reference: TimeRange) -> dict:
        # make the parameters

        thresholds = self.calc_thresholds(reference)
        breakpoint()
        
    def calc_thresholds(self, reference: TimeRange) -> List[str]:

        log('#Thresholds:')

        dest_parameters = self.output_paths['parameters']
        destination = substitute_values(dest_parameters, {'par': 'Qthreshold', 'history_start': reference.start, "history_end": reference.end})
        cases_destinations = [substitute_values(destination, case['tags']) for case in self.cases]

        start_year = reference.start.year
        end_year   = reference.end.year

        input_path = self.input_variable.path

        # check what we need to calculate
        timesteps_to_compute = {case['id']: set() for case in self.cases}
        for doy in range(1,366):
            # get the month and day (this ignores leap years)
            month, day = doy_to_md(doy)
            for case in self.cases:
                this_destination = cases_destinations[case['id']]
                # check if this month and day have already been done: we can put a random year here because %Y is not in destination
                if not check_data(this_destination, datetime(2000, month, day)): 
                    timesteps_to_compute[case['id']].add((month, day))

        for case in self.cases:
            timesteps_done = 365 - len(timesteps_to_compute[case['id']])
            log(f' - case {case["name"]}: {timesteps_done}/365 timesteps already computed.')

        timesteps_to_iterate = set.union(*timesteps_to_compute.values())
        timesteps_to_iterate = list(timesteps_to_iterate)
        timesteps_to_iterate.sort()

        if len(timesteps_to_iterate) == 0:
            return cases_destinations
        
        log(f' #Iterating through {len(timesteps_to_iterate)} timesteps with missing thresholds.')
        for month, day in timesteps_to_iterate:
            log(f'  - {day:02d}/{month:02d}')
            for case in self.cases:
                if (month, day) not in timesteps_to_compute[case['id']]: continue
                this_destination = cases_destinations[case['id']]
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