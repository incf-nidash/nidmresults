import os

from nidmresults._version import __version__

latest_owlfile = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'nidmresults', 'owl',
    "nidm-results_130.owl")