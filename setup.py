from setuptools import setup, find_packages

readme = open('README.rst').read()

reqs = [line.strip() for line in open('requirements.txt').readlines()]
requirements = list(filter(None, reqs))

setup(
    name="nidmresults",
    version="0.2.0",
    author="Camille Maumet",
    author_email="c.m.j.maumet@warwick.ac.uk",
    description=(
        "Export of neuroimaging statistical results using NIDM"
        " as specified at http://nidm.nidash.org/specs/nidm-results.html."),
    license = "MIT",
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
    install_requires=requirements,
)
