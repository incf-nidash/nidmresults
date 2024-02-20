from nidmresults.owl.owl_reader import OwlReader


def test_OwlReader(owl_file):
    """Smoke test of several methods."""
    owl = OwlReader(owl_file=owl_file)
    owl.count_by_namespaces()
