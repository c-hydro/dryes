from typing import Optional, Sequence
import numpy as np
import xarray as xr

from .register_algorithm import as_DRYES_algorithm

@as_DRYES_algorithm
def risk_computation(
        input_data:    Optional[dict[str,xr.DataArray]] = None,
        previous_data: Optional[dict[str,xr.DataArray]] = None,
        static_data:   Optional[dict[str,xr.DataArray]] = None
        ) -> dict[str,xr.DataArray] | tuple[Sequence[str], Sequence[str], Sequence[str], Sequence[str]]:

    input_keys = {'hazard'        : 'dhi',
                  'exposure'      : 'dexpi',
                  'vulnerability' : 'dvi',
                  }

    previous_keys = {}

    static_keys = {'domain': 'Domain'}

    output_keys = ['risk']

    # calling the function without data returns the keys
    if input_data is None and previous_data is None and static_data is None:
        return input_keys, previous_keys, static_keys, output_keys

    # get data from input_data
    hazard, exposure, vulnerability = prepare_risk_inputs(input_data)

    # compute risk
    risk = (hazard * exposure * vulnerability) ** (1/3)

    # create output dictionary
    output = {
        'risk': risk,
    }

    return  output

def prepare_risk_inputs(input_data):

    input_nodata = {k: v.attrs.get('_FillValue') for k, v in input_data.items()}
    input_values = {k: v.values for k, v in input_data.items()}

    hazard = input_values['hazard']
    hazard = np.where(np.isclose(hazard, input_nodata['hazard'], equal_nan=True),  np.nan, hazard)

    exposure = input_values['exposure']
    exposure = np.where(np.isclose(exposure, input_nodata['exposure'], equal_nan=True), np.nan, exposure)

    vulnerability = input_values['vulnerability']
    vulnerability = np.where(np.isclose(vulnerability, input_nodata['vulnerability'], equal_nan=True), np.nan, vulnerability)

    return hazard, exposure, vulnerability