"""
Library Features:

Name:          lib_utils_interp
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210408'
Version:       '1.0.0'
"""
# -------------------------------------------------------------------------------------
# Libraries
import xarray as xr
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to active interpolation method
def active_var_interp(var_attrs, geo_attrs, fields_included=None):

    if fields_included is None:
        fields_included = ['ncols', 'nrows', 'cellsize']

    active_interp = False
    for field_step in fields_included:
        if field_step in list(var_attrs.keys()):
            var_field_value = var_attrs[field_step]
        else:
            var_field_value = None
        if field_step in list(geo_attrs.keys()):
            geo_field_value = geo_attrs[field_step]
        else:
            geo_field_value = None

        if var_field_value is None:
            raise IOError('Attribute "' + field_step + '" in variable data array are not correctly defined')
        if geo_field_value is None:
            raise IOError('Attribute "' + field_step + '" in geo reference data array are not correctly defined')

        if var_field_value != geo_field_value:
            active_interp = True
            break

    return active_interp
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to apply interpolation method
def apply_var_interp(var_obj_in, geo_da_out,
                     dim_name_geo_x='longitude', dim_name_geo_y='latitude',
                     coord_name_geo_x='longitude', coord_name_geo_y='latitude',
                     interp_method='nearest'):

    if not isinstance(var_obj_in, (xr.DataArray, xr.Dataset)):
        raise RuntimeError('Data format for variable not allowed for applying the interpolation method')
    if not isinstance(geo_da_out, xr.DataArray):
        raise RuntimeError('Data format for geographical reference not allowed for applying the interpolation method')

    interp_dict = {dim_name_geo_y: geo_da_out[coord_name_geo_y], dim_name_geo_x: geo_da_out[coord_name_geo_x]}
    var_obj_out = var_obj_in.interp(interp_dict, method=interp_method)

    return var_obj_out

# -------------------------------------------------------------------------------------
