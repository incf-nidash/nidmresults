#!/usr/bin/python
"""Create a database suitable for use with MKDA toolbox.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""


import argparse
import os
import sys

from nidmresults import __version__
from nidmresults.graph import NIDMResults


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
        "-v", "--version", action="version", version=f"{__version__}"
    )

    args = parser.parse_args(argv[1:])

    nidmpacks = args.nidmpacks

    outfile = args.outfile
    if not outfile.endswith(".csv"):
        outfile = f"{outfile}.csv"

    first = True
    con_ids = {None: 0}
    for nidmpack in nidmpacks:
        print(f"Exporting {nidmpack}")

        overwrite = False
        if first:
            overwrite = True
            first = False

        if not os.path.isfile(nidmpack):
            raise Exception(f"Unknown file: {str(nidmpack)}")

        nidmgraph = NIDMResults(nidm_zip=nidmpack)
        con_ids = nidmgraph.serialize(
            outfile,
            "mkda",
            overwrite=overwrite,
            last_used_con_id=max(con_ids.values()),
        )


if __name__ == "__main__":
    main()
