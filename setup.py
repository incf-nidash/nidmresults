from setuptools import setup, find_packages

readme = open('README.rst').read()

reqs = [line.strip() for line in open('requirements.txt').readlines()]
requirements = list(filter(None, reqs))

setup(
    name="nidmresults",
    version="0.3.1",
    author="Camille Maumet",
    author_email="c.m.j.maumet@warwick.ac.uk",
    description=(
        "Export of neuroimaging statistical results using NIDM"
        " as specified at http://nidm.nidash.org/specs/nidm-results.html."),
    license = "MIT",
    scripts=['bin/nidmreader', 'bin/nidm_mkda_convert'],
    url='https://github.com/incf-nidash/nidmresults',
    keywords = "Prov, NIDM, Provenance",
    packages=find_packages(),
    package_dir={
        'nidmresults': 'nidmresults'
    },
    long_description=readme,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
    package_data={'':
                  ['prefixes.csv',
                   'owl/nidm-results_020.owl',
                   'owl/nidm-results_100.owl',
                   'owl/nidm-results_110.owl',
                   'owl/nidm-results_120.owl',
                   'owl/nidm-results_130-rc2.owl']},
    include_package_data=True,
    install_requires=requirements,
)
