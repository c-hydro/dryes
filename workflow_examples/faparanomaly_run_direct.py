from datetime import datetime

from dryes.tools.data import LocalDataset as Local
from dryes.time_aggregation import aggregation_functions as agg
from dryes.post_processing import pp_functions as pp
from dryes.indices import DRYESAnomaly

HOME = '/home/luca/Documents/CIMA_code/tests/VIIRS_test'
INPUT   = f'{HOME}/8d/%Y/%m/%d'
AGGDATA = HOME + '/{agg_fn}/%Y/%m'
PARAMS  = HOME + '/parameters/{history_start:%Y%m%d}-{history_end:%Y%m%d}'
OUTPUT = HOME + '/maps/%Y/%m/%d'

def main():

    VIIRSFAPARAnomaly = DRYESAnomaly(
        index_options = {
            "agg_fn" : {"10d" : agg.weighted_average_of_window(size = 1, unit = 'dekad', input_agg='viirs'),
                        "1m"  : agg.average_of_window(size = 1,  unit = 'months'),
                    },
            "post_fn": {"Sigma2" : pp.gaussian_smoothing(sigma = 2),
                        "Sigma4" : pp.gaussian_smoothing(sigma = 4)}
            },
        io_options = {
            'data_raw': Local(name = 'FAPAR (filtered snow, cloud, pheno)', path = INPUT,
                            file = 'VIIRS-FAPAR_%Y%m%d.tif'), 
            'data':     Local(name = 'FAPAR (filtered and aggregated)', path = AGGDATA,
                            file = 'VIIRS-FAPAR_{agg_fn}_%Y%m%d.tif'),

            # parameters
            'mean': Local(name = 'mean FAPAR',     path = PARAMS,
                        file = 'mean{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            'std':  Local(name = 'gamma.loc (SPI)',   path = PARAMS,
                        file = 'std{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),

            # outputs
            'index': Local(name = 'FAPAR Anomaly', path = OUTPUT,
                           time_signature = 'end+1',
                           file = 'FAPARAnomaly{agg_fn}{post_fn}_%Y%m%d000000.tif',
                           thumbnail = {'colors' :'/home/luca/Documents/viz/FAPAR.txt', 'size': 0.5, 'dpi': 150,
                                        'overlay':'/home/luca/Downloads/Limiti01012024_g/Reg01012024_g/Reg01012024_g_WGS84.shp'})}
)

    current   = (datetime(2024,1,1), datetime(2024,1,31))
    reference = (datetime(2012,1,1), datetime(2020,1,31))
    timesteps_per_year = 36

    #VIIRSFAPARAnomaly.make_parameters(reference, timesteps_per_year)
    VIIRSFAPARAnomaly.make_index(current, reference, timesteps_per_year)

if __name__ == '__main__':
    main()