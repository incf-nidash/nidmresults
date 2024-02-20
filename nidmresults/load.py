"""Load and save NIDM-Results objects."""

import os

from nidmresults.graph import NIDMResults


def load(filename, to_replace=dict()):
    """Load NIDM-Results file given filename.

    Guessing if it is a NIDM-Results pack or a JSON file.

    Parameters
    ----------
    filename : string
       specification of file to load
    Returns
    -------
    nidmres : ``NIDMResults``
       NIDM-Results object
    """
    if not os.path.exists(filename):
        raise IOException("File does not exist: %s" % filename)

    if filename.endswith(".json"):
        raise Exception("Minimal json file: not handled yet")
    elif filename.endswith(".nidm.zip"):
        nidm = NIDMResults.load_from_pack(filename, to_replace=to_replace)
    else:
        raise Exception("Unhandled format " + filename)

    return nidm
