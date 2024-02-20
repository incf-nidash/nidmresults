#!/usr/bin/python
"""Export neuroimaging results created with FSL feat \
   following NIDM-Results specification.

The path to feat directory must be passed as first argument.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""


import argparse
import os
import sys

from nidmresults import __version__
from nidmresults.graph import Graph


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description="NIDM-Results reader.")
    parser.add_argument("nidm_pack", help="Path to NIDM-Results pack.")
    parser.add_argument(
        "--version", action="version", version=f"{__version__}"
    )

    args = parser.parse_args(argv[1:])

    nidm_pack = args.nidm_pack
    if not os.path.isfile(nidm_pack):
        raise Exception(f"Unknown file: {str(nidm_pack)}")

    nidm_graph = Graph(nidm_zip=nidm_pack)
    nidm_graph.parse()

    nidm_graph.get_peaks()
    for peak in nidm_graph.peaks:
        print(peak)

    nidm_graph.get_statistic_maps()
    for stat_map in nidm_graph.stat_maps:
        print(stat_map)


if __name__ == "__main__":
    main()
