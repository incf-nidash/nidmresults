"""
Export neuroimaging results created by neuroimaging software packages
(FSL, AFNI, ...) following NIDM-Results specification.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

from prov.model import ProvBundle, ProvDocument
import rdflib
import os
import datetime
from nidmresults.objects.constants import *
from nidmresults.objects.modelfitting import *
from nidmresults.objects.contrast import *
from nidmresults.objects.inference import *
import uuid
import csv
import tempfile
import zipfile


class NIDMExporter():

    """
    Generic class to parse a result directory to extract the pieces of
    information to be stored in NIDM-Results and to generate a NIDM-Results
    export.
    """

    def __init__(self, version, out_dir, zipped=True):
        out_dirname = os.path.basename(out_dir)
        out_path = os.path.dirname(out_dir)

        # Create output path from output name
        self.zipped = zipped
        if not self.zipped:
            out_dirname = out_dirname+".nidm"
        else:
            out_dirname = out_dirname+".nidm.zip"
        out_dir = os.path.join(out_path, out_dirname)

        # Quit if output path already exists and user doesn't want to overwrite
        # it
        if os.path.exists(out_dir):
            msg = out_dir+" already exists, overwrite?"
            if not raw_input("%s (y/N) " % msg).lower() == 'y':
                quit("Bye.")
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            else:
                os.remove(out_dir)
        self.out_dir = out_dir

        # A temp directory that will contain the exported data
        self.export_dir = tempfile.mkdtemp(prefix="nidm-", dir=out_path)

        if version == "dev":
            self.version = {'major': 10000, 'minor': 0, 'revision': 0,
                            'num': version}
        else:
            major, minor, revision = version.split(".")
            if "-rc" in revision:
                revision, rc = revision.split("-rc")
            else:
                rc = -1
            self.version = {'major': int(major), 'minor': int(minor),
                            'revision': int(revision), 'rc': int(rc),
                            'num': version}

        # Initialise prov document
        self.doc = ProvDocument()
        self._add_namespaces()

        # Initialise rdf document
        self.g = rdflib.Graph()
        self.g.bind("nidm", "http://purl.org/nidash/nidm#")
        self.g.bind("niiri", "http://iri.nidash.org/")
        self.g.bind(
            "crypto",
            "http://id.loc.gov/vocabulary/preservation/cryptographicHashFuncti\
ons#")
        self.g.bind("fsl", "http://purl.org/nidash/fsl#")
        self.g.bind("spm", "http://purl.org/nidash/spm#")
        self.g.bind("dct", "http://purl.org/dc/terms/")
        self.g.bind("obo", "http://purl.obolibrary.org/obo/")
        self.g.bind("dctype", "http://purl.org/dc/dcmitype/")
        self.g.bind("nlx", "http://neurolex.org/wiki/")
        self.g.bind("prov", "http://www.w3.org/ns/prov#")
        self.g.bind("dc", "http://purl.org/dc/elements/1.1/")
        self.g.bind(
            "nfo",
            "http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#")

    def parse(self):
        """
        Parse a result directory to extract the pieces information to be
        stored in NIDM-Results.
        """
        # Methods: find_software, find_model_fitting, find_contrasts and
        # find_inferences should be defined in the children classes and return
        # a list of NIDM Objects as specified in the objects module

        # Object of type Software describing the neuroimaging software package
        # used for the analysis
        self.software = self._find_software()

        # List of objects (or dictionary) of type ModelFitting describing the
        # model fitting step in NIDM-Results (main activity: Model Parameters
        # Estimation)
        self.model_fittings = self._find_model_fitting()

        # Dictionary of (key, value) pairs where where key is a tuple
        # containing the identifier of a ModelParametersEstimation object and a
        # tuple of identifiers of ParameterEstimateMap objects and value is an
        # object of type Contrast describing the contrast estimation step in
        # NIDM-Results (main activity: Contrast Estimation)
        self.contrasts = self._find_contrasts()

        # Inference activity and entities
        # Dictionary of (key, value) pairs where key is the identifier of a
        # ContrastEstimation object and value is an object of type Inference
        # describing the inference step in NIDM-Results (main activity:
        # Inference)
        self.inferences = self._find_inferences()

    def add_object(self, nidm_object):
        """
        Add a NIDMObject to a NIDM-Results export.
        """
        nidm_object.export(self.version)
        # ProvDocument: add object to the bundle
        self.bundle.update(nidm_object.p)
        # RDF document: add object to the main graph
        self.g += nidm_object.g

    def export(self):
        """
        Generate a NIDM-Results export.
        """
        if not os.path.isdir(self.export_dir):
            os.mkdir(self.export_dir)

        # Initialise main bundle
        self._create_bundle(self.version)
        self.add_object(self.software)

        # Add model fitting steps
        for model_fitting in self.model_fittings.values():
            model_fitting.activity.wasAssociatedWith(self.software)
            self.add_object(model_fitting)

        # Add contrast estimation steps
        for (model_fitting_id, pe_ids), contrasts in self.contrasts.items():
            model_fitting = self._get_model_fitting(model_fitting_id)
            for contrast in contrasts:
                contrast.estimation.used(model_fitting.rms_map)
                contrast.estimation.used(model_fitting.mask_map)
                contrast.estimation.wasAssociatedWith(self.software)

                for pe_id in pe_ids:
                    contrast.estimation.used(pe_id)

                self.add_object(contrast)

        # Add inference steps
        for contrast_id, inferences in self.inferences.items():
            contrast = self._get_contrast(contrast_id)

            for inference in inferences:
                if contrast.z_stat_map:
                    used_id = contrast.z_stat_map.id
                else:
                    used_id = contrast.stat_map.id
                inference.inference_act.used(used_id)
                inference.inference_act.wasAssociatedWith(self.software)

                self.add_object(inference)

        # Write-out prov file
        self.save_prov_to_files()

        return self.out_dir

    def _get_model_fitting(self, mf_id):
        """
        Retreive model fitting with identifier 'mf_id' from the list of model
        fitting objects stored in self.model_fitting
        """
        for model_fitting in self.model_fittings.values():
            if model_fitting.activity.id == mf_id:
                return model_fitting
        raise Exception("Model fitting activity with id: " + str(mf_id) +
                        " not found.")

    def _get_contrast(self, con_id):
        """
        Retreive contrast with identifier 'con_id' from the list of contrast
        objects stored in self.contrasts
        """
        for contrasts in self.contrasts.values():
            for contrast in contrasts:
                if contrast.estimation.id == con_id:
                    return contrast
        raise Exception("Contrast activity with id: " + str(con_id) +
                        " not found.")

    def _add_namespaces(self):
        """
        Add namespaces to NIDM document.
        """
        self.doc.add_namespace(NIDM)
        self.doc.add_namespace(NIIRI)
        self.doc.add_namespace(CRYPTO)
        self.doc.add_namespace(DCT)

    def _create_bundle(self, version):
        """
        Initialise NIDM-Results bundle.
        """
        # *** Bundle entity
        bundle_id = NIIRI[str(uuid.uuid4())]

        # provn export
        self.bundle = ProvBundle(identifier=bundle_id)

        self.doc.entity(bundle_id,
                        other_attributes=((PROV['type'], PROV['Bundle'],),
                                          (PROV['label'], "NIDM-Results"),
                                          (PROV['type'], NIDM_RESULTS),
                                          (NIDM_VERSION, version['num']))
                        )

        # ttl export
        bundle_uri = rdflib.URIRef(bundle_id.uri)
        self.g.add((
            bundle_uri, RDF.type, rdflib.URIRef(PROV["Bundle"].uri)))
        self.g.add((
            bundle_uri, RDF.type, rdflib.URIRef(PROV["Entity"].uri)))
        self.g.add((
            bundle_uri, RDF.type, rdflib.URIRef(NIDM_RESULTS.uri)))
        self.g.add((
            bundle_uri, RDFS.label, rdflib.Literal("NIDM-Results")))
        self.g.add((
            bundle_uri,
            rdflib.URIRef(NIDM_VERSION.uri),
            rdflib.Literal(version['num'], datatype=XSD.string)))

        # *** NIDM-Results Export Activity
        if version['num'] not in ["1.0.0", "1.1.0"]:
            self.export = NIDMResultsExport()
            self.export.export(self.version)
            self.doc.update(self.export.p)
            self.g += self.export.g

        # *** bundle was Generated by NIDM-Results Export Activity
        ctime = datetime.datetime.now().time()
        # Qualified generation
        bnode = rdflib.BNode()
        self.g.add((
            bnode, RDF.type, rdflib.URIRef(PROV["Generation"].uri)))
        self.g.add((
            bnode,
            rdflib.URIRef(PROV["atTime"].uri),
            rdflib.Literal(str(ctime), datatype=XSD.dateTime)))
        self.g.add((
            bundle_uri,
            rdflib.URIRef(PROV["qualifiedGeneration"].uri), bnode))

        if version['num'] in ["1.0.0", "1.1.0"]:
            self.doc.wasGeneratedBy(bundle_id, time=str(ctime))
        else:
            # provn
            self.bundle.wasGeneratedBy(self.export.id, time=str(ctime))
            # ttl
            self.g.add((
                bnode,
                rdflib.URIRef(PROV['activity'].uri),
                rdflib.URIRef(self.export.id.uri)))

        # *** NIDM-Results Exporter (Software Agent)
        if version['num'] not in ["1.0.0", "1.1.0"]:
            self.exporter = self._get_exporter()
            self.exporter.export(self.version)
            self.doc.update(self.exporter.p)
            self.g += self.exporter.g

            self.exporter.wasAssociatedWith(self.export.id)
            self.g.add((
                rdflib.URIRef(self.export.id.uri),
                rdflib.URIRef(PROV['wasAssociatedWith'].uri),
                rdflib.URIRef(self.exporter.id.uri)))

    def _get_model_parameters_estimations(self, error_model):
        """
        Infer model estimation method from the 'error_model'. Return an object
        of type ModelParametersEstimation.
        """
        if error_model.dependance == NIDM_INDEPEDENT_ERROR:
            if error_model.variance_homo:
                estimation_method = STATO_OLS
            else:
                estimation_method = STATO_WLS
        else:
            estimation_method = STATO_GLS

        mpe = ModelParametersEstimation(estimation_method, self.software.id)

        return mpe

    def use_prefixes(self, ttl):
        prefix_file = os.path.join(os.path.dirname(__file__), 'prefixes.csv')
        with open(prefix_file, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for alphanum_id, prefix, uri in reader:
                if alphanum_id in ttl:
                    ttl = "@prefix " + prefix + ": <" + uri + "> .\n" + ttl
                    ttl = ttl.replace(alphanum_id, prefix + ":")
        return ttl

    def save_prov_to_files(self, showattributes=False):
        """
        Write-out provn serialisation to nidm.provn.
        """
        self.doc.add_bundle(self.bundle)
        provn_file = os.path.join(self.export_dir, 'nidm.provn')
        provn_fid = open(provn_file, 'w')
        # FIXME None
        provn_fid.write(self.doc.get_provn(4).replace("None", "-"))
        provn_fid.close()

        ttl_file = provn_file.replace(".provn", ".ttl")
        ttl_txt = self.g.serialize(format='turtle')

        ttl_txt = self.use_prefixes(ttl_txt)
        with open(ttl_file, 'w') as ttl_fid:
            ttl_fid.write(ttl_txt)

        # Post-processing
        if not self.zipped:
            # Just rename temp directory to output_path
            os.rename(self.export_dir, self.out_dir)
        else:
            # Create a zip file that contains the content of the temp directory
            os.chdir(self.export_dir)
            zf = zipfile.ZipFile(os.path.join("..", self.out_dir), mode='w')
            try:
                for root, dirnames, filenames in os.walk("."):
                    for filename in filenames:
                        zf.write(os.path.join(filename))
                shutil.rmtree(os.path.join("..", self.export_dir))
            finally:
                zf.close()
                os.chdir("..")

        # ttl_fid = open(ttl_file, 'w');
        # serialization is done in xlm rdf
        # graph = Graph()
        # graph.parse(data=self.doc.serialize(format='rdf'), format="xml")
        # ttl_fid.write(graph.serialize(format="turtle"))
        # ttl_fid.write(self.doc.serialize(format='rdf').
            # replace("inf", '"INF"'))
        # ttl_fid.close()
        # print "provconvert -infile " + provn_file + " -outfile " + ttl_file
        # check_call("provconvert -infile " + provn_file +
        #            " -outfile " + ttl_file, shell=True)
