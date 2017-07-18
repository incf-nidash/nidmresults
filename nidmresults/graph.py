"""
Export neuroimaging results created by neuroimaging software packages
(FSL, AFNI, ...) following NIDM-Results specification.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

from nidmresults.objects.constants import *
from nidmresults.objects.modelfitting import *
from nidmresults.objects.contrast import *
from nidmresults.objects.inference import *

from rdflib.plugins.parsers.notation3 import BadSyntax
import rdflib
import zipfile
import csv


class NIDMResults():
    """
    NIDM-result object containing all metadata and link to image files.
    """

    def __init__(self, nidm_zip=None, rdf_file=None, format="turtle"):
        with zipfile.ZipFile(nidm_zip) as z:
            rdf_data = z.read('nidm.ttl')
        self.rdf_data = rdf_data
        self.study_name = os.path.basename(nidm_zip).replace(".nidm.zip", "")      
        self.format = format
        self.graph = self.parse()
        self.objects = dict()

        self.load_modelfitting()
        self.load_contrasts()
        self.load_inferences()

    @classmethod
    def load_from_pack(klass, nidm_zip):
        nidmr = NIDMResults(nidm_zip=nidm_zip)
        return nidmr

    def get_metadata(self):
        return self.objects

    def parse(self):
        g = rdflib.Graph()
        try:
            g.parse(data=self.rdf_data, format=self.format)
        except BadSyntax:
            raise self.ParseException(
                "RDFLib was unable to parse the RDF file.")
        return g

    def _find_model_fitting(self):
        """
        Parse FSL result directory to retreive model fitting information.
        Return a list of objects of type ModelFitting.
        """
        self.model_fittings = dict()

        for analysis_dir in self.analysis_dirs:

            design_matrix = self._get_design_matrix(analysis_dir)
            data = self._get_data()
            error_model = self._get_error_model()

            rms_map = self._get_residual_mean_squares_map(analysis_dir)
            param_estimates = self._get_param_estimate_maps(analysis_dir)
            mask_map = self._get_mask_map(analysis_dir)
            grand_mean_map = self._get_grand_mean(
                mask_map.file.path, analysis_dir)

            activity = self._get_model_parameters_estimations(error_model)

            # Assuming MRI data
            machine = ImagingInstrument("mri")

            # Group or Person
            if self.version['num'] not in ["1.0.0", "1.1.0", "1.2.0"]:
                if self.first_level:
                    subjects = [Person()]
                else:
                    subjects = list()
                    for group_name, numsub in self.groups:
                        subjects.append(Group(
                            num_subjects=int(numsub), group_name=group_name))
            else:
                subjects = None

            model_fitting = ModelFitting(
                activity, design_matrix, data,
                error_model, param_estimates, rms_map, mask_map,
                grand_mean_map, machine, subjects)

            self.model_fittings[analysis_dir] = model_fitting

        return self.model_fittings

    def get_peaks(self, contrast_name=None):
        """
        Read a NIDM-Results document and return a list of Peaks.
        """
        if contrast_name is not None:
            query_extension = """
?cluster a significant_cluster: .
?peak prov:wasDerivedFrom ?cluster .
?cluster prov:wasDerivedFrom/prov:wasGeneratedBy/prov:used/prov:\
wasGeneratedBy ?conest .
?conmap prov:wasGeneratedBy ?conest .
?conmap a nidm_ContrastMap: .
?conmap nidm_contrastName: ?conname .
            """
        else:
            query_extension = ""

        query = """
prefix prov: <http://www.w3.org/ns/prov#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix nidm_ExcursionSetMap: <http://purl.org/nidash/nidm#NIDM_0000025>
prefix nidm_pValueUncorrected: <http://purl.org/nidash/nidm#NIDM_0000116>
prefix nidm_equivalentZStatistic: <http://purl.org/nidash/nidm#NIDM_0000092>
prefix nidm_coordinateVector: <http://purl.org/nidash/nidm#NIDM_0000086>
prefix nidm_Coordinate: <http://purl.org/nidash/nidm#NIDM_0000015>
prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>

SELECT DISTINCT ?coord_label ?coord_vector ?z ?peak_label ?p_unc
?peak_id ?exc_set_id WHERE {
    ?coord a nidm_Coordinate: ;
        rdfs:label ?coord_label ;
        nidm_coordinateVector: ?coord_vector .
    ?peak_id prov:atLocation ?coord ;
        rdfs:label ?peak_label ;
        nidm_equivalentZStatistic: ?z ;
        nidm_pValueUncorrected: ?p_unc ;
        prov:wasDerivedFrom/prov:wasDerivedFrom ?exc_set_id .
    """ + query_extension + """
}
ORDER BY ?peak_label
        """
        sd = self.graph.query(query)

        peaks = dict()
        if sd:
            for coord_label, coord_vector, z, peak_label, p_unc, peak_id, \
                    exc_set_id in sd:
                peak = Peak(float(z), coord_vector=coord_vector,
                            p_unc=float(p_unc), label=peak_label,
                            coord_label=coord_label,
                            oid=peak_id, exc_set_id=exc_set_id)
                peaks[peak_id] = (peak)
        else:
            print('No peaks found')

        self.objects.update(peaks)
        self.peaks = peaks
        return peaks

    def peak_query(self):
        query = """
        prefix prov: <http://www.w3.org/ns/prov#>
        prefix spm: <http://purl.org/nidash/spm#>
        prefix nidm: <http://purl.org/nidash/nidm#>
        prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        prefix peak: <http://purl.org/nidash/nidm#NIDM_0000062>
        prefix significant_cluster: <http://purl.org/nidash/nidm#NIDM_0000070>
        prefix coordinate: <http://purl.org/nidash/nidm#NIDM_0000086>
        prefix equivalent_zstatistic: <http://purl.org/nidash/\
nidm#NIDM_0000092>
        prefix pvalue_fwer: <http://purl.org/nidash/nidm#NIDM_0000115>
        prefix pvalue_uncorrected: <http://purl.org/nidash/nidm#NIDM_0000116>
        prefix nidm_ContrastMap: <http://purl.org/nidash/nidm#NIDM_0000002>
        prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>

        ####################
        #Peaks Data Query
        ####################
        SELECT ?peak ?xyz ?cluster ?zstat ?pvalfwer ?conname
        WHERE
        { ?peak a peak: .
          ?cluster a significant_cluster: .
          ?peak prov:wasDerivedFrom ?cluster .
          ?peak prov:atLocation ?coordinate .
          ?coordinate coordinate: ?xyz .
          ?peak equivalent_zstatistic: ?zstat .
          OPTIONAL { ?peak pvalue_fwer: ?pvalfwer }.
          ?cluster prov:wasDerivedFrom/prov:wasGeneratedBy/prov:used/prov:\
wasGeneratedBy ?conest .
          ?conmap prov:wasGeneratedBy ?conest .
          ?conmap a nidm_ContrastMap: .
          ?conmap nidm_contrastName: ?conname .
        }
        ORDER BY ?cluster ?peak
        """
        sd = self.graph.query(query)

        peaks = list()
        if sd:
            for peak_id, xyz, cluster, zstat, pfwer, conname in sd:
                local_peak_id = None
                peak = Peak(None, local_peak_id, zstat, 1, cluster_id=cluster,
                            coord_vector=xyz, p_fwer=pfwer, oid=peak_id)
                peaks.append(peak)
        return peaks_df

    def get_coord_spaces(self, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_CoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000016>
prefix nidm_voxelToWorldMapping: <http://purl.org/nidash/nidm#NIDM_0000132>
prefix nidm_voxelUnits: <http://purl.org/nidash/nidm#NIDM_0000133>
prefix nidm_voxelSize: <http://purl.org/nidash/nidm#NIDM_0000131>
prefix nidm_inWorldCoordinateSystem: <http://purl.org/nidash/nidm#NIDM_0000105>
prefix nidm_MNICoordinateSystem: <http://purl.org/nidash/nidm#NIDM_0000051>
prefix nidm_numberOfDimensions: <http://purl.org/nidash/nidm#NIDM_0000112>
prefix nidm_dimensionsInVoxels: <http://purl.org/nidash/nidm#NIDM_0000090>


SELECT ?oid ?label ?vox_to_world ?units ?vox_size ?coordinate_system ?numdim
?dimensions
        WHERE
        {
    """ + oid_var + """ a nidm_CoordinateSpace: ;
    rdfs:label ?label ;
    nidm_voxelToWorldMapping: ?vox_to_world ;
    nidm_voxelUnits: ?units ;
    nidm_voxelSize: ?vox_size ;
    nidm_inWorldCoordinateSystem: ?coordinate_system ;
    nidm_numberOfDimensions: ?numdim ;
    nidm_dimensionsInVoxels: ?dimensions .
    }
        """
        sd = self.graph.query(query)

        coord_spaces = dict()
        if sd:
            for row in sd:
                args = row.asdict()
                if oid is not None:
                    args['oid'] = oid
                coord_spaces[args['oid']] = CoordinateSpace(**args)

        if oid is not None:
            return coord_spaces[oid]
        else:
            return coord_spaces

    def get_subjects(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Subjects.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
SELECT DISTINCT * WHERE {
    """ + oid_var + """ a prov:Person ;
        rdfs:label ?label .
}
        """
        objects = dict()

        arg_list = self.run_query_and_get_args(query, oid)
        for args in arg_list:
            subject = Person(**args)
            objects[args['oid']] = subject

        self.objects.update(objects)
        if oid is not None:
            if oid not in objects:
                raise Exception('No results for query:\n' + query)

            return objects[oid]
        else:
            return objects

    def get_groups(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Groups.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix obo_studygrouppopulation: <http://purl.obolibrary.org/obo/STATO_0000193>
prefix nidm_groupName: <http://purl.org/nidash/nidm#NIDM_0000170>
prefix nidm_numberOfSubjects: <http://purl.org/nidash/nidm#NIDM_0000171>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a obo_studygrouppopulation: ;
        rdfs:label ?label ;
        nidm_groupName: ?group_name ;
        nidm_numberOfSubjects: ?num_subjects .
}
        """
        objects = dict()

        arg_list = self.run_query_and_get_args(query, oid)
        for args in arg_list:
            group = Group(**args)
            objects[args['oid']] = group

        self.objects.update(objects)
        if oid is not None:
            return objects[oid]
        else:
            return objects

    def get_datas(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Data.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_Data: <http://purl.org/nidash/nidm#NIDM_0000169>
prefix obo_studygrouppopulation: <http://purl.obolibrary.org/obo/STATO_0000193>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_Data: ;
        rdfs:label ?label .
        {""" + oid_var + """ prov:wasAttributedTo ?group_id .
        ?group_id a obo_studygrouppopulation: .} UNION
        {""" + oid_var + """ prov:wasAttributedTo ?subject_id .
        ?subject_id a prov:Person .
        } .

}
        """
        objects = dict()
        meta = dict()

        arg_list = self.run_query_and_get_args(query, oid)
        for args in arg_list:
            # FIXME: load from file
            args["grand_mean_scaling"] = None
            args["target"] = None
            args["mri_protocol"] = None

            if "group" in args:
                args["group_or_sub"] = args["group"]
                args.pop("group", None)
            else:
                args["group_or_sub"] = args["subject"]
                args.pop("subject", None)

            # (self, grand_mean_scaling, target, mri_protocol=None, oid=None)
            objects[args['oid']] = Data(**args)

        self.objects.update(objects)

        if oid is not None:
            return objects[oid]
        else:
            return objects

    def get_model_param_estimations(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Model Parameter
        Estimations.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_000005\
6>
prefix nidm_ParameterEstimateMap: <http://purl.org/nidash/nidm#NIDM_0000061>
prefix nidm_Data: <http://purl.org/nidash/nidm#NIDM_0000169>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_ModelParameterEstimation: ;
        rdfs:label ?label ;
        prov:used ?data_id .
    ?data_id a nidm_Data: .
}
        """
        objects = dict()

        arg_list = self.run_query_and_get_args(query, oid)
        for args in arg_list:
            args['estimation_method'] = None
            args['software_id'] = None
            # FIXME: read from file
            objects[args['oid']] = ModelParametersEstimation(**args)
            # (self, estimation_method, software_id):

        self.objects.update(objects)
        if oid is not None:
            return objects[oid]
        else:
            return objects

    def get_param_estimates(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Parameter Estimate
        Maps.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_ParameterEstimateMap: <http://purl.org/nidash/nidm#NIDM_0000061>
prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>
prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_000005\
6>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_ParameterEstimateMap: ;
        rdfs:label ?label ;
        nfo:fileName ?filename ;
        nidm_inCoordinateSpace: ?coord_space_id ;
        crypto:sha512 ?sha ;
        dct:format ?format ;
        prov:wasGeneratedBy ?model_param_estimation_id .
    ?model_param_estimation_id a nidm_ModelParameterEstimation: .
}
        """
        objects = dict()

        arg_list = self.run_query_and_get_args(query, oid)

        pe_num = 1
        for args in arg_list:
            args.pop("format", None)
            args["pe_num"] = pe_num
            args["pe_file"] = None
            pe_num = pe_num + 1
            objects[args['oid']] = ParameterEstimateMap(**args)

        self.objects.update(objects)
        if oid is not None:
            return objects[oid]
        else:
            return objects

    def get_contrast_estimations(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Contrast Estimations.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_ContrastEstimation: <http://purl.org/nidash/nidm#NIDM_0000001>
prefix nidm_ParameterEstimateMap: <http://purl.org/nidash/nidm#NIDM_0000061>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_ContrastEstimation: ;
        rdfs:label ?label ;
        prov:used ?param_estimate_id .
    ?param_estimate_id a nidm_ParameterEstimateMap: .
}
        """
        objects = dict()

        arg_list = self.run_query_and_get_args(query, oid)

        con_num = 1
        for args in arg_list:
            # Assign random contrast number
            args['contrast_num'] = con_num
            con_num = con_num + 1
            # FIXME: deal with more than one param_estimate as input
            objects[args['oid']] = ContrastEstimation(**args)

        self.objects.update(objects)
        if oid is not None:
            return objects[oid]
        else:
            return objects

    def get_statistic_maps(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Statistic Maps.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_StatisticMap: <http://purl.org/nidash/nidm#NIDM_0000076>
prefix nidm_statisticType: <http://purl.org/nidash/nidm#NIDM_0000123>
prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>
prefix nidm_effectDegreesOfFreedom: <http://purl.org/nidash/nidm#NIDM_0000091>
prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>
prefix nidm_errorDegreesOfFreedom: <http://purl.org/nidash/nidm#NIDM_0000093>
prefix obo_tstatistic: <http://purl.obolibrary.org/obo/STATO_0000176>
prefix nidm_ContrastEstimation: <http://purl.org/nidash/nidm#NIDM_0000001>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_StatisticMap: ;
        rdfs:label ?label ;
        prov:atLocation ?location ;
        nidm_statisticType: ?stat_type ;
        nfo:fileName ?filename ;
        dct:format ?format ;
        nidm_contrastName: ?contrast_name ;
        nidm_errorDegreesOfFreedom: ?dof ;
        nidm_effectDegreesOfFreedom: ?effdof ;
        nidm_inCoordinateSpace: ?coord_space_id ;
        crypto:sha512 ?sha ;
        prov:wasGeneratedBy ?contrast_estimation_id .
    ?contrast_estimation_id a nidm_ContrastEstimation: .
}
        """
        objects = dict()

        arg_list = self.run_query_and_get_args(query, oid)
        for args in arg_list:
            stat_map = StatisticMap(**args)
            objects[args['oid']] = stat_map

        self.objects.update(objects)
        if oid is not None:
            return objects[oid]
        else:
            return objects

    def run_query_and_get_args(self, query, oid):
        sd = self.graph.query(query)
        all_args = list()
        if sd:
            for row in sd:
                args = row.asdict()

                for key, value in args.items():

                    # If one the the keys is the id of an object then loads
                    # the corresponding object
                    if key.endswith("_id"):
                        object_name = key.replace("_id", "")
                        method_name = "get_" + object_name + "s"
                        method = getattr(self, method_name)

                        loaded_object = method(value)
                        args[object_name] = loaded_object
                        args.pop(key, None)

                if oid is not None:
                    args['oid'] = oid

                all_args.append(args)
        else:
            raise Exception('No results for query:\n' + query)

        return all_args

    def get_inferences(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Inference activities.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_Inference: <http://purl.org/nidash/nidm#NIDM_0000049>
prefix nidm_ConjunctionInference: <http://purl.org/nidash/nidm#NIDM_0000011>
prefix nidm_hasAltHypothesis: <http://purl.org/nidash/nidm#NIDM_0000097>
prefix nidm_OneTailedTest: <http://purl.org/nidash/nidm#NIDM_0000060>
prefix nidm_StatisticMap: <http://purl.org/nidash/nidm#NIDM_0000076>
prefix spm_PartialConjunctionInferenc: <http://purl.org/nidash/spm#SPM_0000005>

SELECT DISTINCT ?oid ?label ?tail ?stat_map_id WHERE {
    {""" + oid_var + """ a nidm_Inference: .} UNION
    {""" + oid_var + """ a nidm_ConjunctionInference: .} UNION
    {""" + oid_var + """ a spm_PartialConjunctionInferenc: .} .
    """ + oid_var + """ rdfs:label ?label ;
        nidm_hasAltHypothesis: ?tail ;
        prov:used ?stat_map_id .
    ?stat_map_id a nidm_StatisticMap: .
}
        """
        sd = self.graph.query(query)

        inferences = dict()
        if sd:
            for row in sd:
                stat_map = self.get_statistic_maps(row.stat_map_id)
                args = row.asdict()
                args['stat_map'] = stat_map
                args['contrast_num'] = None
                args.pop("stat_map_id", None)
                if oid is not None:
                    args['oid'] = oid

                inferences[args['oid']] = InferenceActivity(**args)

        self.objects.update(inferences)
        if oid is not None:
            return inferences[oid]
        else:
            return inferences

    def get_object(self, klass, oid=None, **kwargs):
        query = klass.get_query(oid)

        sd = self.graph.query(query)

        objects = dict()
        if sd:
            for row in sd:
                argums = row.asdict()
                objects[oid] = klass(**argums, **kwargs)

        self.objects.update(objects)
        if oid is not None:
            if oid not in objects:
                return None
            else:
                return objects[oid]
        else:
            return objects

    def get_contrast_weights(self):
        query = """
        prefix nidm_statisticType: <http://purl.org/nidash/nidm#NIDM_0000123>
        prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>
        prefix obo_contrastweightmatrix: <http://purl.obolibrary.org/obo/STATO_0000323>
        prefix obo_tstatistic: <http://purl.obolibrary.org/obo/STATO_0000176>

        prefix nidm_ContrastEstimation: <http://purl.org/nidash/nidm#NIDM_0000001>

        prefix nidm_ContrastMap: <http://purl.org/nidash/nidm#NIDM_0000002>
        prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>

        SELECT * WHERE {

        ?conw_id a obo_contrastweightmatrix: ;
            rdfs:label ?conw_label ;
            prov:value ?contrast_weights ;
            nidm_statisticType: ?stat_type ; 
            nidm_contrastName: ?contrast_name .

        ?conest_id a nidm_ContrastEstimation: ;
            prov:used ?conw_id .    

        ?conm_id a nidm_ContrastMap: ;
            rdfs:label ?conm_label ;
            prov:atLocation ?conm_location ;
            dct:format ?conm_format ;
            nfo:fileName ?conm_filename ;
            nidm_contrastName: ?contrast_name ;
            nidm_inCoordinateSpace: ?conm_coordspace_id ;
            crypto:sha512 ?conm_sha ;
            prov:wasGeneratedBy ?conest_id .
        }
        """

    def load_modelfitting(self):
        query = """
        prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
        prefix nidm_Data: <http://purl.org/nidash/nidm#NIDM_0000169>
        prefix obo_studygrouppopulation: <http://purl.obolibrary.org/obo/STATO_0000193>
        prefix nidm_ErrorModel: <http://purl.org/nidash/nidm#NIDM_0000023>
        prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_0000056>
        prefix nidm_ResidualMeanSquaresMap: <http://purl.org/nidash/nidm#NIDM_0000066>
        prefix nidm_MaskMap: <http://purl.org/nidash/nidm#NIDM_0000054>
        prefix nidm_GrandMeanMap: <http://purl.org/nidash/nidm#NIDM_0000033>
        prefix nlx_Imaginginstrument: <http://uri.neuinfo.org/nif/nifstd/birnlex_2094>

        SELECT DISTINCT * WHERE {

            ?design_id a nidm_DesignMatrix: .
            OPTIONAL { ?png_id dc:description ?designdesc_id . } .

            ?data_id a nidm_Data: ;
                prov:wasAttributedTo ?machine_id ;
                prov:wasAttributedTo ?person_or_group_id .
            
            {?person_or_group_id a prov:Person .} UNION
            {?person_or_group_id a obo_studygrouppopulation: .} .

            ?machine_id a nlx_Imaginginstrument: .

            ?error_id a nidm_ErrorModel: .

            ?mpe_id a nidm_ModelParameterEstimation: ;
                prov:used ?design_id ;
                prov:used ?data_id ;
                prov:used ?error_id .

            ?rms_id a nidm_ResidualMeanSquaresMap: ;
                nidm_inCoordinateSpace: ?rms_coordspace_id ;
                prov:wasGeneratedBy ?mpe_id .

            ?mask_id a nidm_MaskMap: ;
                nidm_inCoordinateSpace: ?mask_coordspace_id ;
                prov:wasGeneratedBy ?mpe_id .

            ?gm_id a nidm_GrandMeanMap: ;
                nidm_inCoordinateSpace: ?gm_coordspace_id ;
                prov:wasGeneratedBy ?mpe_id .
        }
        """
        sd = self.graph.query(query)

        model_fittings = list()
        if sd:
            for row in sd:
                row_num = 0
                args = row.asdict()

                # TODO: should software_id really be an input?
                activity = self.get_object(ModelParametersEstimation, args['mpe_id'], software_id=None)

                # TODO fill in image_file
                design_matrix_png = None
                if args['png_id'] is not None:
                    design_matrix_png = self.get_object(CoordinateSpace, args['png_id'], export_dir=None)

                design_matrix = self.get_object(DesignMatrix, args['design_id'], matrix=None, image_file=design_matrix_png, export_dir=None)
                data = self.get_object(Data, args['data_id'])
                error_model = self.get_object(ErrorModel, args['error_id'])


                # Find list of model parameter estimate maps
                query_pe_maps = """
                prefix nidm_ParameterEstimateMap: <http://purl.org/nidash/nidm#NIDM_0000061>
                prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>
                
                SELECT DISTINCT * WHERE {
                    ?pe_id a nidm_ParameterEstimateMap: ;
                    nidm_inCoordinateSpace: ?pe_coordspace_id ;
                    prov:wasGeneratedBy <""" + str(args['mpe_id']) + """> .
                }
                """
                param_estimates = list()
                sd_pe_maps = self.graph.query(query_pe_maps)
                if sd_pe_maps:
                    for row_pe in sd_pe_maps:
                        args_pe = row_pe.asdict()
                        pe_map_coordspace = self.get_object(CoordinateSpace, args_pe['pe_coordspace_id'])

                        param_estimates.append(self.get_object(ParameterEstimateMap, args_pe['pe_id'], 
                            coord_space=pe_map_coordspace))

                rms_coord_space = self.get_object(CoordinateSpace, args['rms_coordspace_id'])
                rms_map = self.get_object(ResidualMeanSquares, args['rms_id'], coord_space=rms_coord_space, 
                    export_dir=None)

                mask_coord_space = self.get_object(CoordinateSpace, args['mask_coordspace_id'])
                mask_map = self.get_object(MaskMap, args['mask_id'], coord_space=mask_coord_space, 
                    export_dir=None)

                gm_coord_space = self.get_object(CoordinateSpace, args['gm_coordspace_id'])
                grand_mean_map = self.get_object(GrandMeanMap, args['gm_id'], coord_space=mask_coord_space, 
                    export_dir=None, mask_file=None)

                machine = self.get_object(ImagingInstrument, args['machine_id'])

                subjects = self.get_object(Group, args['person_or_group_id'])

                if subjects is None:
                    # Try loading as a single subject
                    subjects = self.get_object(Person, args['person_or_group_id'])
                
                model_fittings.append(ModelFitting(activity, design_matrix, data, error_model,
                 param_estimates, rms_map, mask_map, grand_mean_map,
                 machine, subjects))

                con_num = row_num + 1        

    def load_contrasts(self):
        query = """
        prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
        prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_0000056>
        prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>
        prefix obo_contrastweightmatrix: <http://purl.obolibrary.org/obo/STATO_0000323>
        prefix nidm_ContrastEstimation: <http://purl.org/nidash/nidm#NIDM_0000001>
        prefix nidm_ContrastMap: <http://purl.org/nidash/nidm#NIDM_0000002>
        prefix nidm_ContrastStandardErrorMap: <http://purl.org/nidash/nidm#NIDM_0000013>
        prefix nidm_ContrastExplainedMeanSquareMap: <http://purl.org/nidash/nidm#NIDM_0000163>
        prefix nidm_StatisticMap: <http://purl.org/nidash/nidm#NIDM_0000076>
        prefix nidm_Inference: <http://purl.org/nidash/nidm#NIDM_0000049>
        prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>

        SELECT DISTINCT * WHERE {

            ?design_id a nidm_DesignMatrix: .

            ?conw_id a obo_contrastweightmatrix: .

            ?conest_id a nidm_ContrastEstimation: ;
                prov:used ?conw_id ;
                prov:used ?design_id .

            ?mpe_id a nidm_ModelParameterEstimation: ;
                prov:used ?design_id .

            ?conm_id a nidm_ContrastMap: ;
                nidm_inCoordinateSpace: ?conm_coordspace_id ;
                nidm_contrastName: ?contrast_name ;
                prov:wasGeneratedBy ?conest_id .

            {?constdm_id a nidm_ContrastStandardErrorMap: .} UNION
            {?constdm_id a nidm_ContrastExplainedMeanSquareMap: .} .

            ?constdm_id nidm_inCoordinateSpace: ?constdm_coordspace_id ;
                prov:wasGeneratedBy ?conest_id .

            ?statm_id a nidm_StatisticMap: ;
                prov:wasGeneratedBy ?conest_id ;
                nidm_inCoordinateSpace: ?statm_coordspace_id .

            ?inf_id a nidm_Inference: ;
                prov:used ?statm_id .

            ?otherstatm_id a nidm_StatisticMap: ;
                prov:wasGeneratedBy ?conest_id ;
                nidm_inCoordinateSpace: ?otherstatm_coordspace_id .


        }
    """
        sd = self.graph.query(query)

        contrasts = dict()
        if sd:
            for row in sd:
                con_num = 0
                args = row.asdict()

                contrast_num = str(con_num)
                contrast_name = args['contrast_name']

                weights = self.get_object(ContrastWeights, args['conw_id'], contrast_num=contrast_num)
                estimation = self.get_object(ContrastEstimation, args['conest_id'], contrast_num=contrast_num)
                contrast_map_coordspace = self.get_object(CoordinateSpace, args['conm_coordspace_id'])
                contrast_map = self.get_object(ContrastMap, args['conm_id'], 
                    coord_space=contrast_map_coordspace, contrast_num=contrast_num, export_dir=None)

                contraststd_map_coordspace = self.get_object(CoordinateSpace, args['constdm_coordspace_id'])

                stderr_or_expl_mean_sq_map = self.get_object(ContrastExplainedMeanSquareMap, args['constdm_coordspace_id'],
                    coord_space=contraststd_map_coordspace, contrast_num=contrast_num, export_dir=None)
                if stderr_or_expl_mean_sq_map is None:
                    # Try loading as a contrast standard map
                    stderr_or_expl_mean_sq_map = self.get_object(ContrastStdErrMap, args['constdm_id'], 
                        coord_space=contraststd_map_coordspace, contrast_num=contrast_num, export_dir=None, is_variance=False, var_coord_space=None)

                stat_map_coordspace = self.get_object(CoordinateSpace, args['statm_coordspace_id'])
                stat_map = self.get_object(StatisticMap, args['statm_id'], coord_space=stat_map_coordspace)

                if args['otherstatm_id'] is not None:
                    zstat_exist = True

                    otherstat_map_coordspace = self.get_object(CoordinateSpace, args['otherstatm_coordspace_id'])
                    otherstat_map = self.get_object(StatisticMap, args['otherstatm_id'], coord_space=stat_map_coordspace)
                    
                if zstat_exist:
                    con = Contrast(contrast_num, args['contrast_name'], weights, estimation,
                          contrast_map, stderr_or_expl_mean_sq_map, stat_map=otherstat_map,
                          z_stat_map=stat_map)
                else:
                    con = Contrast(contrast_num, args['contrast_name'], weights, estimation,
                          contrast_map, stderr_or_expl_mean_sq_map, stat_map=stat_map)

                # Find list of model parameter estimate maps
                query_pe_maps = """
                prefix nidm_ParameterEstimateMap: <http://purl.org/nidash/nidm#NIDM_0000061>
                
                SELECT DISTINCT * WHERE {
                    <""" + str(args['conest_id']) + """> prov:used  ?pe_id .
                    ?pe_id a nidm_ParameterEstimateMap: .
                    
                }
                """
                pe_ids = ()
                sd_pe_maps = self.graph.query(query_pe_maps)
                if sd_pe_maps:
                    for row_pe in sd_pe_maps:
                        args_pe = row_pe.asdict()
                        pe_ids = pe_ids + (args_pe['pe_id'],)

                contrasts[(args['mpe_id'], pe_ids)] = con

                con_num = con_num + 1

    def load_inferences(self):
        query = """
        prefix nidm_ContrastEstimation: <http://purl.org/nidash/nidm#NIDM_0000001>

        prefix nidm_Inference: <http://purl.org/nidash/nidm#NIDM_0000049>
        prefix nidm_HeightThreshold: <http://purl.org/nidash/nidm#NIDM_0000034>
        prefix nidm_ExtentThreshold: <http://purl.org/nidash/nidm#NIDM_0000026>
        prefix nidm_PeakDefinitionCriteria: <http://purl.org/nidash/nidm#NIDM_0000063>
        prefix nidm_ClusterDefinitionCriteria: <http://purl.org/nidash/nidm#NIDM_0000007>
        prefix nidm_DisplayMaskMap: <http://purl.org/nidash/nidm#NIDM_0000020>
        prefix nidm_ExcursionSetMap: <http://purl.org/nidash/nidm#NIDM_0000025>
        prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>
        prefix nidm_SearchSpaceMaskMap: <http://purl.org/nidash/nidm#NIDM_0000068>

        SELECT DISTINCT * WHERE {
            ?con_est_id a nidm_ContrastEstimation: .

            ?inference_id a nidm_Inference: ;
                prov:used/prov:wasGeneratedBy ?con_est_id ;
                prov:used ?height_thresh_id ;
                prov:used ?extent_thresh_id ;
                prov:used ?peak_criteria_id ;
                prov:used ?cluster_criteria_id .

            ?height_thresh_id a nidm_HeightThreshold: .

            ?extent_thresh_id a nidm_ExtentThreshold: .

            ?peak_criteria_id a nidm_PeakDefinitionCriteria: .

            ?cluster_criteria_id a nidm_ClusterDefinitionCriteria: .

            OPTIONAL {
            ?inference_id prov:used ?display_mask_id .
            ?display_mask_id a nidm_DisplayMaskMap: ;
                nidm_inCoordinateSpace: ?disp_coord_space_id . 
            } .

            ?exc_set_id a nidm_ExcursionSetMap: ;
                nidm_inCoordinateSpace: ?excset_coord_space_id ;
                prov:wasGeneratedBy ?inference_id .

            ?search_space_id a nidm_SearchSpaceMaskMap: ;
                nidm_inCoordinateSpace: ?search_space_coord_space_id ;
                prov:wasGeneratedBy ?inference_id .
        }
    """
        sd = self.graph.query(query)

        inferences = dict()
        if sd:
            for row in sd:
                args = row.asdict()
                inference = self.get_object(InferenceActivity, args['inference_id'])

                height_thresh = self.get_object(HeightThreshold, args['height_thresh_id'])
                extent_thresh = self.get_object(ExtentThreshold, args['extent_thresh_id'])
                peak_criteria = self.get_object(PeakCriteria, args['peak_criteria_id'], contrast_num=None)
                cluster_criteria = self.get_object(ClusterCriteria, args['cluster_criteria_id'], contrast_num=None)

                if 'display_mask_id' in args:
                    disp_coordspace = self.get_object(CoordinateSpace, args['disp_coord_space_id'])
                    disp_mask = self.get_object(DisplayMaskMap, args['display_mask_id'], 
                        contrast_num=None, coord_space=disp_coordspace, mask_num=None, export_dir=None)
                else:
                    disp_mask = None

                excset_coordspace = self.get_object(CoordinateSpace, args['excset_coord_space_id'])
                excursion_set = self.get_object(ExcursionSet, args['exc_set_id'], 
                    coord_space=excset_coordspace)

                searchspace_coordspace = self.get_object(CoordinateSpace, args['search_space_coord_space_id'])
                search_space = self.get_object(SearchSpace, args['search_space_id'], 
                    coord_space=searchspace_coordspace, dlh=None, export_dir=None)

                # TODO
                software_id = None

                # Find list of clusters
                query_clusters = """
                prefix nidm_SupraThresholdCluster: <http://purl.org/nidash/nidm#NIDM_0000070>
                
                SELECT DISTINCT * WHERE {
                    ?cluster_id a nidm_SupraThresholdCluster: ;
                    prov:wasDerivedFrom <""" + str(args['exc_set_id']) + """> .
                }
                """
                clusters = list()
                sd_clusters = self.graph.query(query_clusters)
                if sd_clusters:
                    for row_cluster in sd_clusters:
                        args_cl = row_cluster.asdict()

                        # Find list of peaks
                        query_peaks = """
                        prefix nidm_Peak: <http://purl.org/nidash/nidm#NIDM_0000062>
                        
                        SELECT DISTINCT * WHERE {
                            ?peak_id a nidm_Peak: ;
                            prov:wasDerivedFrom <""" + str(args_cl['cluster_id']) + """> .
                        }
                        """
                        peaks = list()
                        sd_peaks = self.graph.query(query_peaks)
                        if sd_peaks:
                            for row_peak in sd_peaks:
                                args_peak = row_peak.asdict()

                                peaks.append(self.get_object(Peak, args_peak['peak_id']))

                        clusters.append(self.get_object(Cluster, args_cl['cluster_id'], peaks=peaks))

                # Dictionary of (key, value) pairs where key is the identifier of a
                # ContrastEstimation object and value is an object of type Inference
                # describing the inference step in NIDM-Results (main activity:
                # Inference)
                inferences[args['con_est_id']] = Inference(inference, height_thresh, extent_thresh,
                    peak_criteria, cluster_criteria, disp_mask, excursion_set,
                    clusters, search_space, software_id)



    def serialize(self, destination, format="mkda", overwrite=False,
                  last_used_con_id=0):
        # We need the peaks, excursion set maps and contrast maps
        self.get_peaks()
        self.get_excursion_set_maps()
        self.get_inferences()
        # self.get_contrast_maps()

        if format == "mkda":
            if not destination.endswith(".csv"):
                destination = destination + ".csv"
            csvfile = destination

            if overwrite:
                add_header = True
                open_mode = 'wb'
            else:
                add_header = False
                open_mode = 'ab'

            with open(csvfile, open_mode) as fid:
                writer = csv.writer(fid, delimiter='\t')
                if add_header:
                    writer.writerow(["9", "NaN", "NaN", "NaN", "NaN", "NaN",
                                     "NaN", "NaN", "NaN"])
                    writer.writerow(
                        ["x", "y", "z", "Study", "Contrast", "N",
                         "FixedRandom", "CoordSys", "Name"])

                # Assumed random effects
                self.FixedRandom = "random"

                # For anything that has a label
                con_ids = dict()
                con_ids[None] = last_used_con_id

                for oid, peak in list(self.get_peaks().items()):
                    exc_set = self.objects[peak.exc_set_id]

                    if exc_set.coord_space.is_mni():
                        space = "MNI"
                    elif exc_set.coord_space.is_talairach():
                        space = "T88"
                    else:
                        raise Exception(
                            "Unrecognised space for " +
                            str(exc_set.coord_space.coordinate_system))

                    stat_map = exc_set.inference.stat_map

                    con_name = stat_map.contrast_name.replace(
                        " ", "_").replace(":", "")

                    # FIXME: need to deal with more than one group
                    self.N = stat_map.contrast_estimation.param_estimate.\
                        model_param_estimation.data.group_or_sub.num_subjects

                    if con_name in con_ids:
                        con_id = con_ids[con_name]
                    else:
                        con_id = max(con_ids.values()) + 1
                        con_ids[con_name] = con_id

                    writer.writerow([
                        peak.coordinate.coord_vector[0],
                        peak.coordinate.coord_vector[1],
                        peak.coordinate.coord_vector[2], self.study_name,
                        con_id,
                        self.N, self.FixedRandom,
                        space, con_name])

                return con_ids

        return con_ids
