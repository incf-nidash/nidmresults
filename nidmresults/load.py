""" Load and save NIDM-Results objects """
from .graph import NIDMResults

import os


def load(filename, to_replace=dict()):
    ''' Load NIDM-Results file given filename, guessing if it is a
    NIDM-Results pack or a JSON file

    Parameters
    ----------
    filename : string
       specification of file to load
    Returns
    -------
    nidmres : ``NIDMResults``
       NIDM-Results object
    '''
    if not os.path.exists(filename):
        raise IOException('File does not exist: %s' % filename)

    if filename.endswith('.json'):
        nidm = NIDMResults.load_from_json(filename, to_replace=to_replace)
    elif filename.endswith('.nidm.zip'):
        nidm = NIDMResults.load_from_pack(filename, to_replace=to_replace)
    else:
        raise Exception('Unhandled format ' + filename)

    return nidm
