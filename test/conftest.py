from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


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


@pytest.fixture()
def to_replace() -> dict[str, str]:
    # Known issues in the NIDM packs
    return {
        " \\ntask": "\\\\n task",
        ';\n    nidm_coordinateVectorInVoxels: "null"^^xsd:string .': ".",
    }


@pytest.fixture()
def owl_file() -> str:
    return str(
        Path(__file__).parent.parent
        / "nidmresults"
        / "owl"
        / "nidm-results_130.owl"
    )
