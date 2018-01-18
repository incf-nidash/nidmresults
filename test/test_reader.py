#!/usr/bin/env python
"""
Test NIDM FSL export tool installation


@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2015
"""
import unittest
from nidmresults.graph import *
from nidmresults.test.test_results_doc import TestResultDataModel
from future.standard_library import hooks
with hooks():
    from urllib.request import urlopen, Request

from nidmresults.owl.owl_reader import OwlReader

import zipfile
import json
# from ddt import ddt, data, unpack
import os
import inspect
import glob
import shutil
from rdflib.compare import isomorphic, graph_diff

import os


# @ddt
class TestReader(unittest.TestCase, TestResultDataModel):

    def setUp(self):
        self.my_execption = ""
        
        owl_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'nidmresults', 'owl', 'nidm-results_130.owl')
        self.owl = OwlReader(owl_file)

        pwd = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))

        # Store test data in a 'data' folder until 'test'
        data_dir = os.path.join(pwd, 'data')

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Collection containing examples of NIDM-Results packs (1.3.0)
        req = Request(
            "http://neurovault.org/api/collections/2210/nidm_results")
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

        # self.packs = glob.glob(os.path.join(data_dir, '*.nidm.zip'))
        self.out_dir = os.path.join(data_dir, 'recomputed')

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        os.mkdir(self.out_dir)

    # @unpack
    # @data({'name': 'excursion set', 'method_name': 'get_excursion_set_maps'},
    #       {'name': 'statistic map', 'method_name': 'get_statistic_maps'})
    def test_read_object(self):
        """
        Test: Check that excursion set can be retreived
        """
        exc = []
        for nidmpack in self.packs:
            print(nidmpack)

            if 'ex_spm_conjunction' in nidmpack:
                export_act_id = 'b8fe52e0f830755481e30d6fae8f6636'
            elif 'ex_spm_contrast_mask' in nidmpack:
                export_act_id = '380868f11cf96af46cc4ea3bf625081f'
            elif 'ex_spm_default' in nidmpack:
                export_act_id = 'e9f2bcf056a679e65838722ab961237c'
            elif 'ex_spm_full_example001' in nidmpack:
                export_act_id = 'fa83bdc818b89b5c70d211d1e03fc7e6'
            elif 'ex_spm_group_ols' in nidmpack:
                export_act_id = '3c3c654f514122eb8007fdf0d2bc03c8'
            elif 'ex_spm_group_wls' in nidmpack:
                export_act_id = 'abb2940ffe4c819e8adb0ab5cd2ecac1'
            elif 'ex_spm_HRF_informed_basis' in nidmpack:
                export_act_id = '85408216d7db19206dcfee6d17c212d3'
            elif 'ex_spm_partial_conjunction' in nidmpack:
                export_act_id = 'f219fd38405d063a1ed632fcb2147443'
            elif 'ex_spm_temporal_derivative' in nidmpack:
                export_act_id = '659c46e2d1c780606dffaab2340049d2'
            elif 'ex_spm_thr_clustfwep05' in nidmpack:
                export_act_id = '67aaee6777a38b8d12724eac8d77eb3e'
            elif 'ex_spm_thr_clustunck10' in nidmpack:
                export_act_id = 'd917e70463d0a73d9dd1b9ebd8edbf12'
            elif 'ex_spm_thr_voxelfdrp05' in nidmpack:
                export_act_id ='824058887a650fffd1658a75fffd3204'
            elif 'ex_spm_thr_voxelfwep05' in nidmpack:
                export_act_id = '5dacfe9b4b68e90ce4283bf549bb2b50'
            elif 'ex_spm_thr_voxelunct4' in nidmpack:
                export_act_id = '3629454df7cfe3fbfb4c3ebc41023e67'
            else:
                export_act_id = ''


            # This is a workaround to avoid confusion between attribute and class uncorrected p-value
            # cf. https://github.com/incf-nidash/nidm/issues/421
            to_replace = {'@prefix nidm_PValueUncorrected: <http://purl.org/nidash/nidm#NIDM_0000160>': 
                          '@prefix nidm_UncorrectedPValue: <http://purl.org/nidash/nidm#NIDM_0000160>',
                          'nidm_PValueUncorrected': 'nidm_UncorrectedPValue',
                          'nidm_PValueUncorrected': 'nidm_UncorrectedPValue',
                          'http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions/': 
                          'http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions#',
                          ' \\ntask': '\\\\n task',
                          'a prov:Generation .': 'a prov:Generation ; prov:activity niiri:' + export_act_id + ' .'}

            nidmres = NIDMResults(nidm_zip=nidmpack, to_replace=to_replace)
            new_name = os.path.join(self.out_dir, os.path.basename(nidmpack))
            nidmres.serialize(new_name)
            print('Serialised to ' + new_name)
            print("----")

            new_nidmres = NIDMResults(nidm_zip=new_name)

            self.compare_full_graphs(nidmres.graph, new_nidmres.graph, self.owl, 
                            include=False, raise_now=False, reconcile=False)

        if self.my_execption:
            raise Exception(self.my_execption)

            # nidm_graph.parse()
            # # exc_sets = nidm_graph.get_excursion_set_maps()

            # method = getattr(nidm_graph, method_name)
            # objects = method()

            # if not objects:
            #     exc.append('No ' + name + ' found for ' + nidmpack)

            # for eid, eobj in objects.items():
            #     with zipfile.ZipFile(nidmpack, 'r') as myzip:
            #         if not str(eobj.file.path) in myzip.namelist():
            #             exc.append(
            #                 'Missing ' + name + ' file for ' + nidmpack)

        # if exc:
        #     raise Exception("\n ".join(exc))

if __name__ == '__main__':
    unittest.main()
