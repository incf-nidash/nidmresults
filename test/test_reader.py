#!/usr/bin/env python
"""Test NIDM FSL export tool installation.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2015
"""
from __future__ import annotations

import os
from pathlib import Path

from nidmresults.graph import NIDMResults
from nidmresults.owl.owl_reader import OwlReader
from nidmresults.test.utils import TestResultDataModel

from .conftest import data_dir, output_dir


def return_list_nidm_zip(path: Path) -> list[str]:
    return [str(x) for x in path.glob("*.nidm.zip")]


def test_read_object(
    download_nidm_results_from_neurovault, to_replace, owl_file
):

    data_model = TestResultDataModel()
    data_model.my_exception = ""
    data_model.out_dir = str(output_dir())

    all_excs = ""
    for nidmpack in return_list_nidm_zip(data_dir()):
        print(nidmpack)

        # Read the NIDM pack
        nidmres = NIDMResults(nidm_zip=nidmpack, to_replace=to_replace)

        # Rewrite the NIDM pack
        new_name = os.path.join(data_model.out_dir, os.path.basename(nidmpack))
        nidmres.serialize(new_name)
        print(f" Serialised to {new_name}")

        # Read the rewritten pack
        new_nidmres = NIDMResults(nidm_zip=new_name)

        # Check equivalence between the two packs (original vs rewritten)
        exc = data_model.compare_full_graphs(
            nidmres.graph,
            new_nidmres.graph,
            owl=OwlReader(owl_file),
            include=False,
            raise_now=False,
            reconcile=False,
        )

        all_excs += exc

    assert not all_excs
