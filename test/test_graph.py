from nidmresults.graph import NIDMResults

from .conftest import data_dir


def test_NIDMResults(download_nidm_results_from_neurovault, to_replace):
    """Smoke test of several methods."""
    nidmpack = data_dir() / "ex_spm_conjunction.nidm.zip"
    nidmres = NIDMResults(nidm_zip=nidmpack, to_replace=to_replace)
    nidmres.get_info()
