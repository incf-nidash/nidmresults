#!/usr/bin/env python
"""Test NIDM FSL export tool installation.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2015
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from nidmresults.graph import NIDMResults
from nidmresults.owl.owl_reader import OwlReader
from nidmresults.test.utils import TestResultDataModel


def owl_file() -> str:
    return str(
        Path(__file__).parent.parent
        / "nidmresults"
        / "owl"
        / "nidm-results_130.owl"
    )


def data_dir() -> Path:
    """Store test data in a 'data' folder until 'test'."""
    tmp = Path(__file__).parent / "data"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def output_dir() -> Path:
    tmp = data_dir() / "recomputed"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(exist_ok=False, parents=True)
    return tmp


def request_nidm_results_from_neurovault(collection: str = None):
    if collection is None:
        # Collection containing examples of NIDM-Results packs (1.3.0)
        collection = "2210"
    req = Request(
        f"http://neurovault.org/api/collections/{collection}/nidm_results"
    )
    rep = urlopen(req)
    response = rep.read()
    data = json.loads(response.decode("utf-8"))
    return data


def return_list_nidm_zip(path: Path) -> list[str]:
    return [str(x) for x in path.glob("*.nidm.zip")]


@pytest.fixture(scope="session")
def download_nidm_results_from_neurovault() -> None:

    data = request_nidm_results_from_neurovault()

    print()

    for nidm_res in data["results"]:

        study = nidm_res["name"]
        nidmpack = data_dir() / f"{study}.zip"

        if nidmpack.exists():
            continue

        url = nidm_res["zip_file"]
        f = urlopen(url)
        print(f"downloading {url} at {nidmpack}")
        with open(nidmpack, "wb") as local_file:
            local_file.write(f.read())


# @pytest.mark.parametrize("nidmpack", return_list_nidm_zip(data_dir()))
def test_read_object(download_nidm_results_from_neurovault):

    data_model = TestResultDataModel()
    data_model.my_exception = ""
    data_model.out_dir = str(output_dir())

    all_excs = ""
    for nidmpack in return_list_nidm_zip(data_dir()):
        print(nidmpack)

        # Known issues in the NIDM packs
        to_replace = {
            " \\ntask": "\\\\n task",
            ';\n    nidm_coordinateVectorInVoxels: "null"^^xsd:string .': ".",
        }

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
            owl=OwlReader(owl_file()),
            include=False,
            raise_now=False,
            reconcile=False,
        )

        all_excs += exc

    assert not all_excs
