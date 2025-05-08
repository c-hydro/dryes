import numpy as np

def calc_anomaly(data: np.ndarray, parameters: dict, method: str = 'empiricalzscore'):
    """
    Calculate the anomaly of the given data using the specified method.
    Parameters:
        data (np.ndarray): The input data for which the anomaly is to be calculated.
        parameters (dict): A dictionary containing the parameters required for the calculation.
            - 'mean' (float): The mean value to be used in the anomaly calculation. This is required.
            - 'std' (float, optional): The standard deviation value, required if the method is 'empiricalzscore'.
        method (str, optional): The method to use for anomaly calculation. Defaults to 'empiricalzscore'.
            Supported methods:
            - 'empiricalzscore': Calculates the empirical z-score anomaly.
            - 'absolutedelta': Calculates the absolute difference anomaly.
            - 'percentdelta': Calculates the percentage difference anomaly.
    Returns:
        np.ndarray: The calculated anomaly values.
    Raises:
        ValueError: If required parameters are missing or if an unknown method is specified.
    """

    mean = parameters.get('mean')
    if mean is None:
        raise ValueError("Mean must be provided for anomaly calculation.")
    
    if method == 'empiricalzscore':
        std = parameters.get('std')
        if std is None:
            raise ValueError("Standard deviation must be provided for empirical z-score method.")
        return (data - mean) / std
    elif method == 'absolutedelta':
        return data - mean
    elif method == 'percentdelta':
        return (data - mean) / mean * 100
    else:
        raise ValueError(f"Unknown method: {method}")

def calc_anomaly_parameters(data: np.ndarray, get_std: bool = True, min_n = 0) -> dict:
    """"
    Calculates the mean and optionally the standard deviation for anomaly calculations 
    from a 3D input array, where the 0-th axis represents time.
    Parameters:
        data (np.ndarray): A 3D numpy array where the 0-th axis is time, and the remaining axes 
                           represent spatial or other dimensions. The data is assumed to correspond 
                           to a single period in the reference period (e.g., all January months in 
                           the reference period).
        get_std (bool): A boolean indicating whether to calculate the standard deviation 
                        in addition to the mean. Defaults to True.
        min_n (int): The minimum number of non-NaN values required to calculate the parameters.
    Returns:
        dict: A dictionary containing the calculated parameters:
              - "mean": The mean values computed along the time axis (0-th axis).
              - "std": The standard deviation values computed along the time axis 
                       (only included if get_std is True).
    """

    parameters = {}
    parameters['mean'] = np.nanmean(data, axis = 0)
    if get_std:
        parameters['std'] = np.nanstd(data, axis = 0)

    n = np.sum(~np.isnan(data), axis=0)
    mask = n > min_n
    for key in parameters.keys():
        parameters[key] = np.where(mask, parameters[key], np.nan)
        
    return parameters