
#!/usr/bin/python
"""
Export neuroimaging results created with FSL feat following NIDM-Results
specification. The path to feat directory must be passed as first argument.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

import os
import argparse
from nidmresults.reader import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NIDM-Results reader.')
    parser.add_argument('rdf_file', help='Path to NIDM-Results file.')

    args = parser.parse_args()

    rdf_file = args.rdf_file
    if not os.path.isfile(rdf_file):
        raise Exception("Unknown file: "+str(rdf_file))

    nidmreader = NIDMReader(rdf_file=rdf_file)

    for peak in nidmreader.peaks:
        print peak

    for stat_map in nidmreader.statistic_maps:
        print stat_map
