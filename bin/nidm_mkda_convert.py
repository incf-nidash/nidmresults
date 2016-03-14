
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
    parser.add_argument('nidmres',
                        help='Path to NIDM-Results archive (.nidm.zip).')

    args = parser.parse_args()

    nidmres = args.nidmres
    if not os.path.isfile(nidmres):
        raise Exception("Unknown file: "+str(nidmres))

    nidmgraph = Graph(nidm_zip=nidmres)
    nidmgraph.serialize('this.csv', "mkda")
