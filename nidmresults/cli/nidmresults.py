#!/usr/bin/python
"""
Return version of nidmresults module

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2016
"""

import sys
import argparse
import nidmresults

def main(argv = sys.argv):
    parser = argparse.ArgumentParser(
        description='NIDM-Results module.')
    parser.add_argument(
        '-v', '--version', action='version',
        version='{version}'.format(version=nidmresults.__version__))

    args = parser.parse_args(argv[1:])


if __name__ == "__main__":
    main()
