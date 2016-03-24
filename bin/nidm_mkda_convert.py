
#!/usr/bin/python
"""
Create a database suitable for use with MKDA toolbox

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

import os
import argparse
from nidmresults.graph import Graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert to MKDA')
    parser.add_argument(
        'nidmpacks',
        help='Path to NIDM-Results packs (.nidm.zip) separated by spaces.',
        nargs="+")

    args = parser.parse_args()

    nidmpacks = args.nidmpacks

    first = True
    for nidmpack in nidmpacks:
        overwrite = False
        if first:
            overwrite = True
            first = False

        if not os.path.isfile(nidmpack):
            raise Exception("Unknown file: "+str(nidmpack))

        nidmgraph = Graph(nidm_zip=nidmpack)
        nidmgraph.serialize('this.csv', "mkda", overwrite=overwrite)
