from datetime import datetime

from dryes.time_aggregation import aggregation_functions as agg
from dryes.post_processing import pp_functions as pp
from dryes.indices import SPI

from dryes.tools.data import LocalDataset as Local

def main():

    HOME = '/home/luca/Documents/CIMA_code/tests/SPI_test'
    INPUT = f'{HOME}/1d/%Y/%m'

    AGGDATA = HOME + '/{agg_fn}m/%Y/%m/%d/'
    PARAMS  = HOME + '/parameters/{history_start:%Y%m%d}-{history_end:%Y%m%d}'
    OUTPUT = f'{HOME}/maps/%Y/%m/%d/'

    spi_mcm_ita = SPI(
        index_options = {
            "agg_fn" : {"1"  : agg.sum_of_window(size = 1,  unit = 'months'),
                        "3"  : agg.sum_of_window(size = 3,  unit = 'months'),
                        #"6"  : agg.sum_of_window(size = 6,  unit = 'months'),
                        #"12" : agg.sum_of_window(size = 12, unit = 'months')
                    },
            #"distribution" : {"gamma":"gamma", "normal":"normal", "pearson3":"pearson3"},
            #"pval_threshold" : {"t05": 0.05, "t10": 0.1},
            #"post_fn": {"Sigma2" : pp.gaussian_smoothing(sigma = 2),
            #            "Sigma4" : pp.gaussian_smoothing(sigma = 4)}
            },
        io_options = {
            'data_raw': Local(name = 'precipitation', path = INPUT,
                            file = 'volta_final_CHIRPS_%Y%m%d.tif'), 
            'data':     Local(name = 'precipitation (aggregated)', path = AGGDATA,
                            file = 'volta_final_CHIRPS_{agg_fn}_%Y%m%d.tif'),

            # parameters
            'gamma.a':       Local(name = 'gamma.a (SPI)',     path = PARAMS,
                                   file = 'gamma/a/aAgg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            'gamma.loc':     Local(name = 'gamma.loc (SPI)',   path = PARAMS,
                                   file = 'gamma/loc/locAgg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            'gamma.scale':   Local(name = 'gamma.scale (SPI)', path = PARAMS,
                                   file = 'gamma/scale/scaleAgg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            # 'normal.loc':    Local(name = 'normal.loc (SPI)',   path = PARAMS,
            #                     file = 'normal/loc/locAgg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            # 'normal.scale':  Local(name = 'normal.scale (SPI)', path = PARAMS,
            #                     file = 'normal/scale/scaleAgg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            # 'pearson3.skew': Local(name = 'pearson3.a (SPI)',     path = PARAMS,
            #                     file = 'pearson3/skew/skewAgg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            # 'pearson3.loc':  Local(name = 'pearson3.loc (SPI)',   path = PARAMS,
            #                     file = 'pearson3/loc/locAgg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            # 'pearson3.scale':Local(name = 'pearson3gamma.scale (SPI)', path = PARAMS,
            #                     file = 'pearson3/scale/scaleAgg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            'prob0':         Local(name = 'prob_0 (SPI)',      path = PARAMS,
                                   file = 'prob0/prob0Agg{agg_fn}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
            
            # outputs
            'index': Local(name = 'Standardised Precipitation Index (SPI)', 
                           path = OUTPUT, file = 'SPI{agg_fn}_%Y%m%d000000.tif',
                           time_signature = 'end+1')}
    )
    
    current   = (datetime(2020,1,1), datetime(2020,12,31))
    reference = (datetime(2010,1,1), datetime(2019,12,31))
    timesteps_per_year = 36

    #spi_mcm_ita.make_parameters(reference, timesteps_per_year)
    spi_mcm_ita.make_index(current, reference, timesteps_per_year)

if __name__ == '__main__':
    main()

