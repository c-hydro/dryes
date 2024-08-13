from datetime import datetime

from dryes.tools.data import LocalDataset as Local
from dryes.indices import HWI

import os

HOME = '/home/luca/Documents/CIMA_code/tests/ERA5/ITA_HCWI'

DATA       = os.path.join(HOME, 'data', '%Y', '%m', '%d')
PARAMETERS = os.path.join(HOME, 'parameters')
OUTPUT     = os.path.join(HOME, 'output')

def main():
# # Define the index
    ITA_HWI = HWI(
        io_options = {
            'data_raw': Local(name = 'Daily temperature', path = DATA, file = 'ERA5_{Ttype}2m_temperature_%Y%m%d.tif'),

            # parameters
            'threshold':  Local(name = 'Temperature 90th percentile',         path = os.path.join(PARAMETERS, 'thresholds'),
                                file = 'T{Ttype}/T{Ttype}90_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),

            'intensity':  Local(name = 'Heatwave intensity (raw)',    path = os.path.join(PARAMETERS, 'ddi', '%Y', '%m'),
                                file = 'raw_intensity_%Y%m%d.tif'),
            'duration':   Local(name = 'Heatwave duration',      path = os.path.join(PARAMETERS, 'ddi', '%Y', '%m'),
                                file = 'duration_%Y%m%d.tif'),
            'interval':   Local(name = 'Time since last heatwaves', path = os.path.join(PARAMETERS, 'ddi', '%Y', '%m'),
                                file = 'interval_%Y%m%d.tif'),
                                
            # output
            'index': Local(name = 'Heatwave intensity', path = OUTPUT, file = 'HWIntensity_%Y%m%d.tif')}
    )

    current   = (datetime(1990,1,1), datetime(1990,1,31)) #(datetime(2020,1,1), datetime(2024,7,31))
    reference = (datetime(1990,1,1), datetime(1991,12,31)) #datetime(2020,12,31))
    timesteps_per_year = 36

    #ITA_HWI.make_parameters(reference, timesteps_per_year)
    ITA_HWI.make_index(current, reference, timesteps_per_year)

if __name__ == '__main__':
    main()