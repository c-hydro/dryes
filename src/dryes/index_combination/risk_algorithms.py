from typing import Optional, Sequence
import numpy as np
import xarray as xr

from .register_algorithm import as_DRYES_algorithm

@as_DRYES_algorithm(
        input  = ['hazard', 'exposure', 'vulnerability'],
        output = ['risk']
)
def risk_computation(
        input_data: dict[str,xr.DataArray],
        exponent  : float = 1/3) -> dict[str,xr.DataArray]:

    xr_template = input_data['hazard'].copy()
    xr_template.attrs = {}

    # change no_data values to nan for all input data and make into np.arrays
    for k, v in input_data.items():
        nd = v.attrs.get('_FillValue')
        if nd is not None:
            input_data[k] = np.where(np.isclose(v, nd, equal_nan=True), np.nan, v)

    # compute risk
    risk = (input_data['hazard'] * input_data['exposure'] * input_data['vulnerability']) ** (exponent)
    risk = xr_template.copy(data=risk.astype(np.float32))
    risk.attrs['_FillValue'] = np.nan

    # create output dictionary
    output = {
        'risk': risk,
    }

    return  output