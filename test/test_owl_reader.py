import pytest
import rdflib

from nidmresults.owl.owl_reader import OwlReader


def test_OwlReader(owl_file):
    """Smoke test of several methods."""
    owl = OwlReader(owl_file=owl_file)

    owl.count_by_namespaces()
    owl.get_class_names()
    owl.get_sub_class_names()


@pytest.mark.parametrize(
    "owl_term, expected",
    [
        ("Organization", "Agent"),
        ("Collection", "Entity"),
        ("Usage", "Influence"),
    ],
)
def test_get_prov_class(owl_term, expected, owl_file):
    """Smoke test of several methods."""
    owl = OwlReader(owl_file=owl_file)
    result = owl.get_prov_class(
        owl_term=rdflib.term.URIRef(f"http://www.w3.org/ns/prov#{owl_term}")
    )
    assert result == rdflib.term.URIRef(
        f"http://www.w3.org/ns/prov#{expected}"
    )
