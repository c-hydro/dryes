from datetime import datetime
import functools

from DRYES.time_aggregation import aggregation_functions as agg
from DRYES.post_processing import pp_functions as pp
from DRYES.indices import DRYESStandardisedIndex

from DRYES.io import LocalIOHandler as Local

HOME = '/home/drought/DRYES/analyses/luca_workspace/tests/SPI/Italia'
INPUT = f'{HOME}/data/%Y/%m/'

ARCHIVE = f'{HOME}/archive'
AGGDATA = f'{ARCHIVE}/data/%Y/%m/%d/'
PARAMS  = ARCHIVE + '/parameters/{history_start:%Y%m%d}-{history_end:%Y%m%d}'
OUTPUT = f'{ARCHIVE}/maps/%Y/%m/%d/'


SPI = DRYESStandardisedIndex(
    index_options = {
        "agg_fn" : {"1"  : agg.sum_of_window(size = 1,  unit = 'months'),
                    "3"  : agg.sum_of_window(size = 3,  unit = 'months'),
                    #"6"  : agg.sum_of_window(size = 6,  unit = 'months'),
                    #"12" : agg.sum_of_window(size = 12, unit = 'months')
                   },
        "distribution" : "gamma",
        "pval_threshold" : {"t05": 0.05, "t10": 0.1},
        "post_fn": {"Sigma2" : pp.gaussian_smoothing(sigma = 2),
                    "Sigma4" : pp.gaussian_smoothing(sigma = 4)}
        },
    io_options = {
        'data_raw': Local(name = 'precipitation', path = INPUT,
                          file = 'PrecipMCM_nonnegative_%Y%m%d.tif'), 
        'data':     Local(name = 'precipitation (aggregated)', path = AGGDATA,
                          file = 'PrecipMCM{agg_fn}Months_%Y%m%d.tif'),

        # parameters
        'gamma.a':     Local(name = 'gamma.a (SPI)',     path = PARAMS,
                             file = 'a/aAgg{agg_fn}{pval_threshold}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
        'gamma.loc':   Local(name = 'gamma.loc (SPI)',   path = PARAMS,
                             file = 'loc/locAgg{agg_fn}{pval_threshold}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
        'gamma.scale': Local(name = 'gamma.scale (SPI)', path = PARAMS,
                             file = 'scale/scaleAgg{agg_fn}{pval_threshold}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
        'prob0':       Local(name = 'prob_0 (SPI)',      path = PARAMS,
                             file = 'prob0/prob0Agg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
        
        # outputs
        'log'  : Local(name = 'log SPI', path = HOME, file =  'log.txt'),
        'index': Local(name = 'Standardised Precipitation Index (SPI)', 
                       path = OUTPUT, file = 'SPI{agg_fn}{pval_threshold}{post_fn}_%Y%m%d000000.tif')}
)

SPI.compute(current   = (datetime(2020,1,1), datetime(2020,1,31)),
            reference = (datetime(2010,1,1), datetime(2022,12,31)),
            timesteps_per_year = 12)