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
import json
from ddt import ddt, data, unpack
import os
import inspect


@ddt
class TestReader(unittest.TestCase):

    def setUp(self):
        pwd = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))

        # Store test data in a 'data' folder until 'test'
        data_dir = os.path.join(pwd, 'data')

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Collection containing examples of NIDM-Results packs (1.3.0)
        req = Request(
            "http://neurovault.org/api/collections/1692/nidm_results")
        rep = urlopen(req)

        response = rep.read()
        data = json.loads(response.decode('utf-8'))

        # Download the NIDM-Results packs from NeuroVault if not available
        # locally
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

    @unpack
    @data({'name': 'excursion set', 'method_name': 'get_excursion_set_maps'},
          {'name': 'statistic map', 'method_name': 'get_statistic_maps'})
    def test_read_object(self, name, method_name):
        """
        Test: Check that excursion set can be retreived
        """
        exc = []
        for nidmpack in self.packs:
            nidm_graph = Graph(nidm_zip=nidmpack)
            nidm_graph.parse()
            # exc_sets = nidm_graph.get_excursion_set_maps()

            method = getattr(nidm_graph, method_name)
            objects = method()

            if not objects:
                exc.append('No ' + name + ' found for ' + nidmpack)

            for eid, eobj in objects.items():
                with zipfile.ZipFile(nidmpack, 'r') as myzip:
                    if not str(eobj.file.path) in myzip.namelist():
                        exc.append(
                            'Missing ' + name + ' file for ' + nidmpack)

        if exc:
            raise Exception("\n ".join(exc))

if __name__ == '__main__':
    unittest.main()
