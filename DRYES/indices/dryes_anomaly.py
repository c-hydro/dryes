
from .dryes_index import DRYESIndex

from ..time_aggregation import aggregation_functions as agg
from ..lib.time import TimeRange


class DRYESAnomaly(DRYESIndex):

    default_options = {
        'agg_fn' : {'Agg1': agg.average_of_window(1, 'months')},
        'type'   : 'empiricalzscore'
    }

    parameters = ('mean', 'std')

    def calc_parameters(self, reference: TimeRange) -> dict:
        breakpoint()
        pass