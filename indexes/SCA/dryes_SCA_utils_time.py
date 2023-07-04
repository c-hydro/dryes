#######################################################################################
# Libraries
import logging
import pandas as pd
import datetime

from copy import deepcopy

import matplotlib.pylab as plt
#######################################################################################

# -------------------------------------------------------------------------------------
# Method to enforce midnight
def enforce_midnight(time_arg):

    time_arg_pd = pd.to_datetime(time_arg)
    time_arg_now_pd_truncated = datetime.date(time_arg_pd.year, time_arg_pd.month, time_arg_pd.day)
    time_arg_now = time_arg_now_pd_truncated.strftime("%Y-%m-%d %H:%M")

    return time_arg_now

