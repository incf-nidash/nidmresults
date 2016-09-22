#!/usr/bin/env python
"""
Test NIDM FSL export tool installation


@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2015
"""
import unittest
from nidmresults.graph import *
from future.standard_library import hooks
with hooks():
    from urllib.request import urlopen, Request

import zipfile
import tempfile
import json
import os


class TestReader(unittest.TestCase):

    def setUp(self):
        data_cfg = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'test_data.json')
        with open(data_cfg) as data_file:
            data_loc = json.load(data_file)

        if not data_loc['location']:
            # Store test data in a temporary folder
            data_loc['location'] = tempfile.mkdtemp()

        data_dir = data_loc['location']

        req = Request(
            "http://neurovault.org/api/collections/1692/nidm_results")
        rep = urlopen(req)

        response = rep.read()
        data = json.loads(response.decode('utf-8'))

        self.packs = list()
        for nidm_res in data["results"]:
            url = nidm_res["zip_file"]
            study = nidm_res["name"]

            nidmpack = os.path.join(data_dir, study + ".zip")
            if not os.path.isfile(nidmpack):
                f = urlopen(url)
                print("downloading " + url + " at " + nidmpack)
                with open(nidmpack, "wb") as local_file:
                    local_file.write(f.read())
            self.packs.append(nidmpack)

    def test_exc_set(self):
        """
        Test: Check that excursion set can be retreived
        """
        exc = []
        for nidmpack in self.packs:
            nidm_graph = Graph(nidm_zip=nidmpack)
            nidm_graph.parse()
            exc_sets = nidm_graph.get_excursion_set_maps()

            if not exc_sets:
                exc.append('No excursion set found for ' + nidmpack)

            for eid, eobj in exc_sets.items():
                with zipfile.ZipFile(nidmpack, 'r') as myzip:
                    if not str(eobj.file.path) in myzip.namelist():
                        exc.append(
                            'Missing excursion set file for ' + nidmpack)

        if exc:
            raise Exception("\n ".join(exc))

if __name__ == '__main__':
    unittest.main()
