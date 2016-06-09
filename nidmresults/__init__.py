import os
from pkg_resources import get_distribution

latest_owlfile = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'nidmresults', 'owl',
    "nidm-results_130.owl")

__version__ = get_distribution('nidmresults').version
