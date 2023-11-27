from datetime import datetime, timedelta
import numpy as np
import xarray as xr

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

    def make_parameters(self, history: TimeRange):
        # for the LFI, we need to do this separately from the superclass, because parameters work a little differently
        # the threshold is calculated from the superclass
        self.parameters = ('Qthreshold',)
        super().make_parameters(history)

        # then we need to calcualte the deficit for each timestep and the lambda

    def calc_parameters(self, dates: List[datetime],
                        par_and_cases: dict[str:List[int]],
                        reference: TimeRange) -> dict[str:dict[int:xr.DataArray]]:
        """
        Calculates the parameters for the LFI. This will only do the threshold calculation.
        par_and_cases is a dictionary with the following structure:
        {par: [case1, case2, ...]}
        indicaing which cases from self.cases['opt'] need to be calculated for each parameter.

        The output is a dictionary with the following structure:
        {par: {case1: parcase1, case2: parcase1, ...}}
        where parcase1 is the parameter par for case1 as a xarray.DataArray.
        """

        input_path = self.input_variable.path
        output = {'Qthreshold': {}} # this is the output dictionary

        # loop over all cases, let's do this in a smart way so that we don't have to load the data multiple times
        window_cases = {}
        for case in self.cases['opt']:
            if case['id'] not in par_and_cases['Qthreshold']: continue
            # get the window
            window = case['options']['thr_window']
            # add the case to the dictionary
            window_cases[window] = window_cases.get(window, []) + [case['id']]
        
        for window, cases in window_cases.items():
            halfwindow = np.floor(window/2)
            # get all dates within the window of this month and day in the reference period
            all_dates = []
            for date in dates:
                this_date = date - timedelta(days = halfwindow)
                while this_date <= date + timedelta(days = halfwindow):
                    if this_date >= reference.start and this_date <= reference.end:
                        all_dates.append(this_date)
                    this_date += timedelta(days = 1)
            
            # if there are no dates, skip
            if len(all_dates) == 0: continue
            # get the data for these dates
            data = [get_data(input_path, date) for date in all_dates]
            data_template = data[0].copy()
            data = np.stack(data, axis = 0)

            # calculate the thresholds for each case that has this window size
            for case in cases:
                threshold_quantile = self.cases['opt'][case]['options']['threshold']
                threshold_data = np.quantile(data, threshold_quantile, axis = 0)
                # save the threshold
                threshold = data_template.copy(data = threshold_data)
                output['Qthreshold'][case] = threshold

        return output

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