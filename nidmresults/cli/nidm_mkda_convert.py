#!/usr/bin/python
"""Create a database suitable for use with MKDA toolbox.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

from nidmresults import __version__
from nidmresults.graph import Graph


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(
        description="Convert a set of NIDM-Results packs to an MKDA-compliant \
        csv file."
    )
    parser.add_argument(
        "-o", "--outfile", help="Name of the csv file.", default="mkda.csv"
    )
    parser.add_argument(
        "nidmpacks",
        help="Path to NIDM-Results packs (.nidm.zip) separated by spaces.",
        nargs="+",
    )
    parser.add_argument(
        "--version", action="version", version="{version}".format(version=__version__)
    )

    args = parser.parse_args(argv[1:])

    nidmpacks = args.nidmpacks

    outfile = args.outfile
    if not outfile.endswith(".csv"):
        outfile = outfile + ".csv"

    first = True
    con_ids = dict()
    con_ids[None] = 0
    for nidmpack in nidmpacks:
        print("Exporting " + nidmpack)

        overwrite = False
        if first:
            overwrite = True
            first = False

        if not os.path.isfile(nidmpack):
            raise Exception("Unknown file: " + str(nidmpack))

        nidmgraph = Graph(nidm_zip=nidmpack)
        con_ids = nidmgraph.serialize(
            outfile, "mkda", overwrite=overwrite, last_used_con_id=max(con_ids.values())
        )


if __name__ == "__main__":
    main()
