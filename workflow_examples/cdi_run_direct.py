from dryes.indices import DRYESCombinedIndicator
from dryes.tools.data import LocalDataset as Local
import os

HOME = '/home/luca/Documents/CIMA_code/tests/DRYES_CDI'

INPUT_PATH  = os.path.join(HOME, 'input',  '%Y', '%m', '%d')
COUNT_PATH  = os.path.join(HOME, 'counts', '%Y', '%m', '%d')
OUTPUT_PATH = os.path.join(HOME, 'output', '%Y', '%m', '%d')

VIZ = '/home/luca/Documents/viz'

io_options = {
    'spi1' :  Local(path = INPUT_PATH,  filename = 'spi1_cdi_%Y%m%d.tif',  time_signature =  'start'),
    'spi3' :  Local(path = INPUT_PATH,  filename = 'spi3_cdi_%Y%m%d.tif',  time_signature =  'start'),
    'sma'  :  Local(path = INPUT_PATH,  filename = 'sma_cdi_%Y%m%d.tif',   time_signature =  'start'),
    'fapar':  Local(path = INPUT_PATH,  filename = 'fapar_cdi_%Y%m%d.tif', time_signature =  'start'),
    'cdi_p' : Local(path = OUTPUT_PATH, filename = 'cdi_%Y%m%d.tif',       time_signature =  'start'),
    'count_fapar_recovery': Local(path = COUNT_PATH, filename = 'count_count_fapar_recovery_%Y%m%d.tif', time_signature = 'start'),
    'count_sma_recovery'  : Local(path = COUNT_PATH, filename = 'count_count_sma_recovery_%Y%m%d.tif',   time_signature = 'start'),
    'cases' : Local(path = OUTPUT_PATH, filename = 'cases_%Y%m%d.tif', time_signature = 'start'),
    'index' : Local(path = OUTPUT_PATH, filename = 'cdi_%Y%m%d.tif',   time_signature = 'start',
                    thumbnail = {
                        'colors' : {''     : os.path.join(VIZ, 'CDI.txt'),
                                    'cdi_p': os.path.join(VIZ, 'CDI.txt'),
                                    'spi1' : os.path.join(VIZ, 'SPI.txt'),
                                    'spi3' : os.path.join(VIZ, 'SPI.txt'),
                                    'sma'  : os.path.join(VIZ, 'SMA.txt'),
                                    'fapar': os.path.join(VIZ, 'FAPAR.txt')},
                        'overlay' : os.path.join(VIZ, 'countries/countries.shp',),
                        'destination' : os.path.join(HOME, 'thumbnails', '%Y%m%d.pdf'),
                    }),
}

test_index = DRYESCombinedIndicator(io_options = io_options)
test_index.make_index(current = ('2024-07-21', '2024-07-31'), timesteps_per_year = 36)