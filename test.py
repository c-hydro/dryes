from datetime import datetime
from DRYES.data_sources.cds import efas
import os

from DRYES.data_processes import *
from DRYES.data_store import DRYESDataStore
from DRYES.lib.log import setup_logging

grid_file = '/home/drought/share/DRYES/data/Italia/static/ET/MCM_mask.tif'

output_path = '/home/drought/DRYES/analyses/luca_workspace/tests/CDS_SM_lisflood/Italia/output'
output_static = os.path.join(output_path, 'static')
output_dynamic = os.path.join(output_path, '{time:%Y}/{time:%m}/{time:%d}')

log_file = '/home/drought/DRYES/analyses/luca_workspace/tests/CDS_SM_lisflood/Italia/log.txt'

if __name__ == '__main__':

    setup_logging(log_file)

    grid       = Grid(grid_file)
    date_start = datetime.datetime(1992, 1, 1)
    date_end   = datetime.datetime(1993, 1, 1)
    timesteps  = TimeRange(date_start, date_end)

    field_capacity = efas.EFASDownloader('field_capacity').average_soil_layers(save_before=True)
    field_capacity.make_local(
                grid = grid, timesteps = None,
                name = 'LISFLOOD',
                path = output_static)

    wilting_point = efas.EFASDownloader('wilting_point').average_soil_layers(save_before=True)
    wilting_point.make_local(
                grid = grid, timesteps = None,
                name = 'LISFLOOD',
                path = output_static)

    soil_moisture = efas.EFASDownloader('volumetric_soil_moisture').average_soil_layers(save_before=True)
    soil_moisture.make_local(
                grid = grid, timesteps = timesteps,
                name = 'LISFLOOD',
                path = output_dynamic)




    # soil_moisture_downloader = efas.EFASDownloader('volumetric_soil_moisture')
    # soil_moisture = soil_moisture_downloader.get_data(grid, datetime(1992, 1, 1))

    # timeagg = partial(aggregate_average, aggtime = 1, aggunit = 'month')
    # timesteps = create_timesteps(date_start, date_end, n_intervals=12)

    # i = 1
    # t0 = time.time()
    # for data in aggregate_time(soil_moisture, timesteps, timeagg):
    #     print(f'timestep {i} of {len(timesteps)}')
    #     i += 1
    #     save_to_geotiff(data, os.path.join(output_path, output_pattern))
    #     t1 = time.time()
    #     print(f'elapsed time: {t1 - t0}')
    #     t0 = time.time()
