from typing import Optional, Sequence
import numpy as np
import xarray as xr

from ..core.combined_indicators import cdi_v4, cdi_norec, prepare_cdi_inputs
from .register_algorithm import as_DRYES_algorithm

@as_DRYES_algorithm(
        input    = ['spi1', 'spi3', 'fapar', 'sma'],
        previous = ['cdi_p', 'count_fapar_recovery', 'count_sma_recovery'],
        static   = ['domain'],
        output   = ['cdi', 'count_sma_recovery', 'count_fapar_recovery', 'cases']
)
def cdi_jrc(input_data:    dict[str,xr.DataArray],
            previous_data: Optional[dict[str,xr.DataArray]] = None,
            static_data:   Optional[dict[str,xr.DataArray]] = None,
            ) -> dict[str,xr.DataArray]:

    # get the xr_template from the spi1 (the only mandatory input)
    xr_template = input_data['spi1'].copy()
    xr_template.attrs = {}

    # prepare the inputs
    input_data = {k:v.values for k,v in input_data.items() if v is not None}
    previous_data = {k:v.values for k,v in previous_data.items() if v is not None}
    all_inputs = prepare_cdi_inputs(**input_data, **previous_data)

    # get the domain
    domain = static_data['domain'].values

    # calculate the cdi
    cdi, count_sma_recovery, count_fapar_recovery, cases = cdi_v4(*all_inputs, domain=domain)

    # create the output as xarray DataArrays
    cdi = xr_template.copy(data=cdi.astype(np.uint8))
    cdi.attrs['_FillValue']  = 8

    count_sma_recovery = xr_template.copy(data=count_sma_recovery.astype(np.uint8))
    count_sma_recovery.attrs['_FillValue'] = 255

    count_fapar_recovery = xr_template.copy(data=count_fapar_recovery.astype(np.uint8))
    count_fapar_recovery.attrs['_FillValue'] = 255

    cases = xr_template.copy(data=cases.astype(np.int16))
    cases.attrs['_FillValue'] = 999

    # create the output dictionary
    output = {
        'cdi':                  cdi,
        'count_sma_recovery':   count_sma_recovery,
        'count_fapar_recovery': count_fapar_recovery,
        'cases':                cases
    }

    return  output

@as_DRYES_algorithm(
        input    = ['spi1', 'spi3', 'fapar', 'sma'],
        previous = ['cdi_p'],
        static   = ['domain'],
        output   = ['cdi', 'cases']
)
def cdi_jrc_norecovery(
            input_data:    dict[str,xr.DataArray],
            previous_data: Optional[dict[str,xr.DataArray]] = None,
            static_data:   Optional[dict[str,xr.DataArray]] = None
            ) -> dict[str,xr.DataArray]:

    # get the xr_template from the spi1 (the only mandatory input)
    xr_template = input_data['spi1'].copy()
    xr_template.attrs = {}

    # prepare the inputs
    all_inputs = prepare_cdi_inputs(**input_data, **previous_data)

    # get the domain
    domain = static_data['domain'].values

    # calculate the cdi
    cdi, cases = cdi_norec(*all_inputs[:5], domain=domain)
    
    # create the output as xarray DataArrays
    cdi = xr_template.copy(data=cdi)
    cdi.attrs['_FillValue']  = 8

    cases = xr_template.copy(data=cases)
    cases.attrs['_FillValue'] = 999

    return  {'cdi': cdi, 'cases': cases}