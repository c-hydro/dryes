from datetime import datetime

from dryes.io import LocalIOHandler as Local
from dryes.indices import DRYESLFI


HOME = '/home/drought/DRYES/analyses/luca_workspace/tests/LFI/PO'

LOG_FILE = f'{HOME}/log.txt'

GRID_FILE = f'{HOME}/PoNetwork_mask.tif'

DATA = f'{HOME}/output/%Y/%m/%d'
ARCHIVE = f'{HOME}/archive'

# # Define the index
PoBasinLFI = DRYESLFI(
    index_options = {"thr_quantile": {"Thr005":.05, "Thr010": 0.1, "Thr020": 0.2}},
    io_options = {
        'data': Local(name = 'discharge', path = DATA, file = 'discharge_%Y%m%d.tif'),

        # parameters
        'Qthreshold': Local(name = 'Qthreshold (LFI)',     path = f'{ARCHIVE}/parameters/Qthreshold',
                            file = 'Qthreshold_{thr_quantile}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%m%d.tif'),
        'lambda':     Local(name = 'lambda (LFI)',   path = f'{ARCHIVE}/parameters/lambda',
                            file = 'lambda_{thr_quantile}_{history_start:%Y%m%d}-{history_end:%Y%m%d}.tif'),
        'Ddeficit':   Local(name = 'Streamflow deficit (LFI)', path = f'{ARCHIVE}/parameters/ddi/%Y/%m/%d/',
                            file = 'Ddeficit_{thr_quantile}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%Y%m%d.tif'),
        'duration':   Local(name = 'Drought duration (LFI)',      path = f'{ARCHIVE}/parameters/ddi/%Y/%m/%d/',
                            file = 'duration_{thr_quantile}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%Y%m%d.tif'),
        'interval':   Local(name = 'interdrought interval (LFI)',      path = f'{ARCHIVE}/parameters/ddi/%Y/%m/%d/',
                            file = 'interval_{thr_quantile}_{history_start:%Y%m%d}-{history_end:%Y%m%d}_%Y%m%d.tif'),
        # outputs
        'log'  : Local(name = 'log LFI', path = HOME, file =  'log.txt'),
        'index': Local(name = 'Low Flow Index (LFI)', path = f'{ARCHIVE}/maps/%Y/%m/%d/',
                       file = 'LFI{thr_quantile}_%Y%m%d000000.tif')}
)

PoBasinLFI.compute(current   = (datetime(2020,9,1), datetime(2022,9, 1)),
                   reference = (datetime(2009,9,1), datetime(2020,8,31)),
                   timesteps_per_year = 36)