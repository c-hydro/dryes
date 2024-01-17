from datetime import datetime

from dryes.io import LocalIOHandler as Local
from dryes.time_aggregation import aggregation_functions as agg
from dryes.post_processing import pp_functions as pp
from dryes.indices import DRYESAnomaly

INPUT   = '/home/drought/share/DRYES/data/Italia/output/FPAR/VNP15A2H/%Y/%m/%d'
HOME = '/home/drought/DRYES/analyses/luca_workspace/tests/VIIRS-FAPAR/Italia'

ARCHIVE = f'{HOME}/archive'

AGGDATA = f'{ARCHIVE}/data/%Y/%m/%d/'
PARAMS  = ARCHIVE + '/parameters/{history_start:%Y%m%d}-{history_end:%Y%m%d}'
OUTPUT = f'{ARCHIVE}/maps/%Y/%m/%d/'

VIIRSFAPARAnomaly = DRYESAnomaly(
    index_options = {
        "agg_fn" : {"Agg1"  : agg.average_of_window(size = 1,  unit = 'months'),
                    "Agg3"  : agg.average_of_window(size = 3,  unit = 'months'),
                    "Agg6"  : agg.average_of_window(size = 6,  unit = 'months'),
                    "Agg12" : agg.average_of_window(size = 12, unit = 'months')
                   },
        "type"   : "empiricalzscore",
        "post_fn": {"Sigma2" : pp.gaussian_smoothing(sigma = 2),
                    "Sigma4" : pp.gaussian_smoothing(sigma = 4)}
        },
    io_options = {
        'data_raw': Local(name = 'FAPAR (filtered snow, cloud, pheno)', path = INPUT,
                          file = 'VIIRS-FPAR_%Y%m%d_filtered-clouds-snow-pheno.tif'), 
        'data':     Local(name = 'FAPAR (filtered and aggregated)', path = AGGDATA,
                          file = 'FAPAR{agg_fn}_%Y%m%d.tif'),

        # parameters
        'mean': Local(name = 'mean FAPAR',     path = PARAMS,
                      file = 'mean{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
        'std':  Local(name = 'gamma.loc (SPI)',   path = PARAMS,
                      file = 'std{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),

        # outputs
        'log'  : Local(name = 'log FAPAR', path = HOME, file =  'log.txt'),
        'index': Local(name = 'FAPAR Anomaly', 
                       path = OUTPUT, file = 'FAPARAnomaly{agg_fn}{post_fn}_%Y%m%d000000.tif')}
)

VIIRSFAPARAnomaly.compute(
            current   = (datetime(2012,1,1), datetime(2023,1,31)),
            reference = (datetime(2012,1,1), datetime(2022,12,31)),
            timesteps_per_year = 12)