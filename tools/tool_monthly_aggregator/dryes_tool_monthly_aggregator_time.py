# -------------------------------------------------------------------------------------
# Libraries
import logging
import pandas as pd
from datetime import date
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to enforce start or end of month wherever needed
def check_end_start_month(time_arg, start_month=None, end_month=None):

    time_arg_dt = pd.to_datetime(time_arg)
    if start_month:
        if time_arg_dt.is_month_start is False:
            time_arg_dt = time_arg_dt.replace(day=1)
            logging.warning(' ==> ' + time_arg + ' enforced to first day of the month!')
            logging.warning(' ==> This script only supports fixed monthly steps!')

    if end_month:
        if time_arg_dt.is_month_end is False:
            time_arg_dt = time_arg_dt.replace(day=time_arg_dt.days_in_month)
            logging.warning(' ==> ' + time_arg + ' enforced to last day of the month!')
            logging.warning(' ==> This script only supports fixed monthly steps!')

    # convert back to string
    time_arg = time_arg_dt.strftime("%Y-%m-%d %H:%M")
    return time_arg
# -------------------------------------------------------------------------------------

