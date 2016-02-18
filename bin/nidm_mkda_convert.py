
#!/usr/bin/python
"""
Create a database suitable for use with MKDA toolbox

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

import os
import argparse
from nidmresults.reader import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert to MKDA')
    parser.add_argument('nidmres',
                        help='Path to NIDM-Results archive (.nidm.zip).')

    args = parser.parse_args()

    nidmres = args.nidmres
    if not os.path.isfile(nidmres):
        raise Exception("Unknown file: "+str(nidmres))

    nidmreader = NIDMReader(nidm_zip=nidmres)
    nidmreader.write_mkda_database(csvfile='this.csv')
