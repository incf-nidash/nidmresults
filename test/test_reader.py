#!/usr/bin/env python
"""Test NIDM FSL export tool installation.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2015
"""
import glob
import inspect
import json

# from ddt import ddt, data, unpack
import os
import shutil
import unittest

from future.standard_library import hooks

from nidmresults.graph import *
from nidmresults.owl.owl_reader import OwlReader
from nidmresults.test.test_results_doc import TestResultDataModel

with hooks():
    from urllib.request import Request, urlopen


# @ddt
class TestReader(unittest.TestCase, TestResultDataModel):

    def setUp(self):
        self.my_execption = ""

        owl_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "nidmresults",
            "owl",
            "nidm-results_130.owl",
        )
        self.owl = OwlReader(owl_file)

        pwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        # Store test data in a 'data' folder until 'test'
        data_dir = os.path.join(pwd, "data")

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Collection containing examples of NIDM-Results packs (1.3.0)
        req = Request("http://neurovault.org/api/collections/2210/nidm_results")
        rep = urlopen(req)

        response = rep.read()
        data = json.loads(response.decode("utf-8"))

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

        self.packs = glob.glob(os.path.join(data_dir, "*.nidm.zip"))
        self.out_dir = os.path.join(data_dir, "recomputed")

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        os.mkdir(self.out_dir)

    def test_read_object(self):
        """Round-trip test.

        Check that we can read all NIDM packs,
        rewrite them and get the same pack again
        """
        all_excs = ""
        for nidmpack in self.packs:
            print(nidmpack)

            # Known issues in the NIDM packs
            to_replace = {
                " \\ntask": "\\\\n task",
                ';\n    nidm_coordinateVectorInVoxels: "null"^^xsd:string .': ".",
            }

            # Read the NIDM pack
            nidmres = NIDMResults(nidm_zip=nidmpack, to_replace=to_replace)

            # Rewrite the NIDM pack
            new_name = os.path.join(self.out_dir, os.path.basename(nidmpack))
            nidmres.serialize(new_name)
            print("Serialised to " + new_name)
            print("----")

            # Read the rewritten pack
            new_nidmres = NIDMResults(nidm_zip=new_name)

            # Check equivalence between the two packs (original vs rewritten)
            exc = self.compare_full_graphs(
                nidmres.graph,
                new_nidmres.graph,
                self.owl,
                include=False,
                raise_now=False,
                reconcile=False,
            )

            all_excs = all_excs + exc

        if all_excs:
            raise Exception(all_excs)


if __name__ == "__main__":
    unittest.main()
