from subprocess import check_call


def test_cli():
    """Check CLI entrypoints."""
    check_call("nidm_mkda_convert -v", shell=True)
    check_call("nidmreader -v", shell=True)
    check_call("nidmresults -v", shell=True)
