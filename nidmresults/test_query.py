# import prov.model as prov

# document = prov.ProvDocument()

# document.set_default_namespace('http://example.org/0/')
# document.add_namespace('ex1', 'http://example.org/1/')
# document.add_namespace('ex2', 'http://example.org/2/')

# document.entity('e001')

# bundle = document.bundle('e001')
# bundle.set_default_namespace('http://example.org/2/')
# bundle.entity('e001')

# print(document.get_provn())

# print('---')

# print(document.serialize(format='rdf', rdf_format='turtle')) # =>

# {"prefix": {"default": "http://example.org/0/", "ex2": "http://example.org/2/", "ex1": "http://example.org/1/"}, "bundle": {"e001": {"prefix": {"default": "http://example.org/2/"}, "entity": {"e001": {}}}}, "entity": {"e001": {}}}

import zipfile
import rdflib

zip_path = '/Users/cmaumet/Projects/Data_sharing/dev/nidmresults_dev/test/data/recomputed/ex_spm_conjunction.nidm.zip'

# Load the turtle file
with zipfile.ZipFile(zip_path) as z:
    rdf_data = z.read('nidm.ttl')
graph = rdflib.Graph()
try:
    graph.parse(data=rdf_data, format="turtle")
except BadSyntax:
    raise ParseException(
        "RDFLib was unable to parse the RDF file.")

# oid_var = '<http://iri.nidash.org/474d479d-6f7a-43f5-94f9-4f2353e56d55>'

# query = """
#         prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
#         prefix spm_SPMsDriftCutoffPeriod: <http://purl.org/nidash/spm#SPM_0000001>
#         prefix fsl_driftCutoffPeriod: <http://purl.org/nidash/fsl#FSL_0000004>

#         SELECT DISTINCT * WHERE {
#             [] a nidm_DesignMatrix: ;
#                 nidm_hasDriftModel: """ + oid_var + """ .



#             {""" + oid_var + """ spm_SPMsDriftCutoffPeriod: ?parameter .} UNION
#             {""" + oid_var + """ fsl_driftCutoffPeriod: ?parameter .} .
#         }
#         """
query = """
        prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
        prefix nidm_hasDriftModel: <http://purl.org/nidash/nidm#NIDM_0000088>
        prefix nidm_Data: <http://purl.org/nidash/nidm#NIDM_0000169>
        prefix nidm_ErrorModel: <http://purl.org/nidash/nidm#NIDM_0000023>
        prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_0000056>
        prefix nidm_ResidualMeanSquaresMap: <http://purl.org/nidash/nidm#NIDM_0000066>
        prefix nidm_MaskMap: <http://purl.org/nidash/nidm#NIDM_0000054>
        prefix nidm_GrandMeanMap: <http://purl.org/nidash/nidm#NIDM_0000033>
        prefix nlx_Imaginginstrument: <http://uri.neuinfo.org/nif/nifstd/birnlex_2094>
        prefix nlx_MagneticResonanceImagingScanner: <http://uri.neuinfo.org/nif/nifstd/birnlex_2100>
        prefix nlx_PositronEmissionTomographyScanner: <http://uri.neuinfo.org/nif/nifstd/ixl_0050000>
        prefix nlx_SinglePhotonEmissionComputedTomographyScanner: <http://uri.neuinfo.org/nif/nifstd/ixl_0050001>
        prefix nlx_MagnetoencephalographyMachine: <http://uri.neuinfo.org/nif/nifstd/ixl_0050002>
        prefix nlx_ElectroencephalographyMachine: <http://uri.neuinfo.org/nif/nifstd/ixl_0050003>
        prefix nidm_ReselsPerVoxelMap: <http://purl.org/nidash/nidm#NIDM_0000144>

        SELECT DISTINCT * WHERE {

            ?design_id a nidm_DesignMatrix: .
            OPTIONAL { ?design_id dc:description ?png_id . } .
            OPTIONAL { ?design_id nidm_hasDriftModel: ?drift_model_id . } .

            ?data_id a nidm_Data: ;
                prov:wasAttributedTo ?machine_id .
            
            {?machine_id a nlx_Imaginginstrument: .} UNION
            {?machine_id a nlx_MagneticResonanceImagingScanner: .} UNION
            {?machine_id a nlx_PositronEmissionTomographyScanner: .} UNION
            {?machine_id a nlx_SinglePhotonEmissionComputedTomographyScanner: .} UNION
            {?machine_id a nlx_MagnetoencephalographyMachine: .} UNION
            {?machine_id a nlx_ElectroencephalographyMachine: .}

            ?error_id a nidm_ErrorModel: .

            ?mpe_id a nidm_ModelParameterEstimation: ;
                prov:used ?design_id ;
                prov:used ?data_id ;
                prov:used ?error_id .

            ?rms_id a nidm_ResidualMeanSquaresMap: ;
                nidm_inCoordinateSpace: ?rms_coordspace_id ;
                prov:wasGeneratedBy ?mpe_id .

            
        }
        """

sd = graph.query(query)

if not sd:
    print('No results found for query')
else:
    print(sd)