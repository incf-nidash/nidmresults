"""
Export neuroimaging results created by neuroimaging software packages
(FSL, AFNI, ...) following NIDM-Results specification.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

import nidmresults
from nidmresults.objects.constants import *
from nidmresults.objects.modelfitting import *
from nidmresults.objects.contrast import *
from nidmresults.objects.inference import *
from nidmresults.exporter import NIDMExporter

from rdflib.plugins.parsers.notation3 import BadSyntax
import rdflib
import zipfile
import csv


class NIDMResults():
    """
    NIDM-result object containing all metadata and link to image files.
    """

    def __init__(self, nidm_zip=None, rdf_file=None):

        self.study_name = os.path.basename(nidm_zip).replace(".nidm.zip", "")      

        # Load the turtle file
        with zipfile.ZipFile(nidm_zip) as z:
            rdf_data = z.read('nidm.ttl')
        self.graph = rdflib.Graph()
        try:
            self.graph.parse(data=rdf_data, format="turtle")
        except BadSyntax:
            raise self.ParseException(
                "RDFLib was unable to parse the RDF file.")

        self.objects = dict()

        # Query the RDF document 
        self.software = self.load_software()
        self.model_fittings = self.load_modelfitting()
        self.contrasts = self.load_contrasts()
        self.inferences = self.load_inferences()

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

    def get_object(self, klass, oid=None, **kwargs):
        query = klass.get_query(oid)

        sd = self.graph.query(query)

        objects = dict()
        if sd:
            for row in sd:
                argums = row.asdict()
                objects[oid] = klass(oid=oid, **argums, **kwargs)

        self.objects.update(objects)
        if oid is not None:
            if oid not in objects:
                return None
            else:
                return objects[oid]
        else:
            return objects

    def load_software(self):
        query = """
        prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_0000056>

        SELECT DISTINCT * WHERE {

            ?ni_software_id a prov:SoftwareAgent .

            ?mpe_id a nidm_ModelParameterEstimation: ;
                prov:wasAssociatedWith ?ni_software_id .
        }
        """
        sd = self.graph.query(query)

        model_fittings = list()
        if sd:
            for row in sd:
                args = row.asdict()
                software = self.get_object(NeuroimagingSoftware, args['ni_software_id'])
        
        return software
        

    def load_modelfitting(self):
        query = """
        prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
        prefix nidm_hasDriftModel: <http://purl.org/nidash/nidm#NIDM_0000088>
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
            OPTIONAL { ?design_id dc:description ?png_id . } .
            OPTIONAL { ?design_id nidm_hasDriftModel: ?drift_model_id . } .

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
                activity = self.get_object(ModelParametersEstimation, args['mpe_id'], software_id=self.software.id)

                if 'png_id' in args:
                    design_matrix_png = self.get_object(Image, args['png_id'])
                else:
                    design_matrix_png = None

                if 'drift_model_id' in args:
                    drift_model = self.get_object(DriftModel, args['drift_model_id'])
                else:
                    drift_model = None

                design_matrix = self.get_object(DesignMatrix, args['design_id'], matrix=None, 
                    image_file=design_matrix_png, drift_model=drift_model)
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
                            coord_space=pe_map_coordspace, pe_num=None))

                rms_coord_space = self.get_object(CoordinateSpace, args['rms_coordspace_id'])
                rms_map = self.get_object(ResidualMeanSquares, args['rms_id'], coord_space=rms_coord_space)

                mask_coord_space = self.get_object(CoordinateSpace, args['mask_coordspace_id'])
                mask_map = self.get_object(MaskMap, args['mask_id'], coord_space=mask_coord_space)

                gm_coord_space = self.get_object(CoordinateSpace, args['gm_coordspace_id'])
                grand_mean_map = self.get_object(GrandMeanMap, args['gm_id'], coord_space=mask_coord_space, 
                    mask_file=None)

                machine = self.get_object(ImagingInstrument, args['machine_id'])

                subjects = self.get_object(Group, args['person_or_group_id'])

                if subjects is None:
                    # Try loading as a single subject
                    subjects = self.get_object(Person, args['person_or_group_id'])
                
                model_fittings.append(ModelFitting(activity, design_matrix, data, error_model,
                 param_estimates, rms_map, mask_map, grand_mean_map,
                 machine, subjects))

                con_num = row_num + 1        

        return model_fittings

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
                    coord_space=contrast_map_coordspace, contrast_num=contrast_num)

                contraststd_map_coordspace = self.get_object(CoordinateSpace, args['constdm_coordspace_id'])

                stderr_or_expl_mean_sq_map = self.get_object(ContrastExplainedMeanSquareMap, args['constdm_coordspace_id'],
                    coord_space=contraststd_map_coordspace, contrast_num=contrast_num)
                if stderr_or_expl_mean_sq_map is None:
                    # Try loading as a contrast standard map
                    stderr_or_expl_mean_sq_map = self.get_object(ContrastStdErrMap, args['constdm_id'], 
                        coord_space=contraststd_map_coordspace, contrast_num=contrast_num, is_variance=False, var_coord_space=None)

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

        if not contrasts:
            raise Exception('No contrast found')

        return contrasts

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
                        contrast_num=None, coord_space=disp_coordspace, mask_num=None)
                else:
                    disp_mask = None

                excset_coordspace = self.get_object(CoordinateSpace, args['excset_coord_space_id'])
                excursion_set = self.get_object(ExcursionSet, args['exc_set_id'], 
                    coord_space=excset_coordspace)

                searchspace_coordspace = self.get_object(CoordinateSpace, args['search_space_coord_space_id'])
                search_space = self.get_object(SearchSpace, args['search_space_id'], 
                    coord_space=searchspace_coordspace, dlh=None)

                # TODO
                software_id = self.software.id

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

        return inferences



    def serialize(self, destination, format="nidm", overwrite=False,
                  last_used_con_id=0):

        if format == "nidm":
            exporter = NIDMExporter(version="1.3.0", out_dir=destination)
            exporter.model_fittings = self.model_fittings
            exporter.contrasts = self.contrasts
            exporter.inferences = self.inferences
            exporter.exporter = ExporterSoftware('nidmresults', nidmresults.__version__)
            exporter.software = self.software
            exporter.export()

        elif format == "mkda":
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
