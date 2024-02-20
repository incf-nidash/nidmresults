"""Export neuroimaging results created by neuroimaging software packages \
(FSL, AFNI, ...) following NIDM-Results specification.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

import collections
import csv
import warnings
import zipfile

import rdflib
from rdflib.plugins.parsers.notation3 import BadSyntax

from nidmresults.exporter import NIDMExporter
from nidmresults.objects.constants_rdflib import *
from nidmresults.objects.contrast import *
from nidmresults.objects.inference import *
from nidmresults.objects.modelfitting import *

# from rdflib.term import Literal


class NIDMResults:
    """NIDM-result object containing all metadata and link to image files."""

    def __init__(self, nidm_zip=None, rdf_file=None, workaround=False, to_replace=dict()):
        self.study_name = os.path.basename(nidm_zip).replace(".nidm.zip", "")
        self.zip_path = nidm_zip

        # Load the turtle file
        with zipfile.ZipFile(self.zip_path, "r") as z:
            rdf_data = z.read("nidm.ttl")
        rdf_data = rdf_data.decode()

        # Exporter-version-specific fixes in the RDF
        rdf_data = self.fix_for_specific_versions(rdf_data, to_replace)

        # Parse turtle into RDF graph
        self.graph = self.parse(rdf_data)

        self.objects = dict()
        self.info = None

        # Query the RDF document and create the objects
        self.software = self.load_software()
        (self.bundle, self.exporter, self.export_act, self.export_time) = (
            self.load_bundle_export()
        )
        self.model_fittings = self.load_modelfitting()
        self.contrasts = self.load_contrasts(workaround=workaround)
        self.inferences = self.load_inferences()

    def fix_for_specific_versions(self, rdf_data, to_replace):
        """Fix of the RDF before loading the graph.

        All of these are workaround
        to circumvent known issues of the SPM and FSL exporters.
        """
        # Load the graph as is so that we can query
        g = self.parse(rdf_data)

        # Retrieve the exporter name and version
        query = """
prefix nidm_spm_results_nidm: <http://purl.org/nidash/nidm#NIDM_0000168>
prefix nidm_nidmfsl: <http://purl.org/nidash/nidm#NIDM_0000167>
prefix nidm_softwareVersion: <http://purl.org/nidash/nidm#NIDM_0000122>

SELECT DISTINCT ?type ?version ?exp_act WHERE {
    {?exporter a nidm_nidmfsl: .} UNION {?exporter a nidm_spm_results_nidm: .}.
    ?exporter a ?type ;
        nidm_softwareVersion: ?version .

    ?exp_act prov:wasAssociatedWith ?exporter .

    FILTER ( ?type NOT IN (prov:SoftwareAgent, prov:Agent))
}
        """

        sd = g.query(query)
        objects = dict()
        if sd:
            for row in sd:
                argums = row.asdict()
                if argums["type"] == NIDM_SPM_RESULTS_NIDM and (
                    argums["version"].eq("12.6903") or argums["version"].eq("12.575ac2c")
                ):
                    warnings.warn(
                        "Applying fixes for SPM exporter " + str(argums["version"])
                    )
                    # crypto namespace inconsistent with NIDM-Results spec
                    to_replace[
                        (
                            "http://id.loc.gov/vocabulary/preservation/"
                            + "cryptographicHashFunctions/"
                        )
                    ] = (
                        "http://id.loc.gov/vocabulary/preservation/"
                        + "cryptographicHashFunctions#"
                    )
                    # Missing 'activity' attribute in qualified Generation
                    to_replace["a prov:Generation ."] = (
                        "a prov:Generation ; prov:activity <"
                        + str(argums["exp_act"])
                        + "> ."
                    )

        # Avoid confusion between attribute and
        # class uncorrected p-value
        # cf. https://github.com/incf-nidash/nidm/issues/421
        to_replace[
            (
                "@prefix nidm_PValueUncorrected: "
                + "<http://purl.org/nidash/nidm#NIDM_0000160>"
            )
        ] = (
            "@prefix nidm_UncorrectedPValue: "
            + "<http://purl.org/nidash/nidm#NIDM_0000160>"
        )
        to_replace["nidm_PValueUncorrected"] = "nidm_UncorrectedPValue"

        if to_replace is not None:
            for to_rep, replacement in to_replace.items():
                rdf_data = rdf_data.replace(to_rep, replacement)

        return rdf_data

    @classmethod
    def load_from_pack(klass, nidm_zip, workaround=False, to_replace=dict()):
        nidmr = NIDMResults(
            nidm_zip=nidm_zip, workaround=workaround, to_replace=to_replace
        )
        return nidmr

    def get_info(self):
        if self.info is None:
            self.info = collections.OrderedDict()

            # TODO: here we assume that there is a single mpe activity per
            # nidm-results pack, this should
            # be stated explicitly in the spec?
            if len(self.model_fittings) > 1:
                raise Exception(
                    "Can't handle NIDM-Results packs with \
                    multiple model parameter estimation activities"
                )

            self.info["NeuroimagingAnalysisSoftware_type"] = self.software.name
            self.info["NeuroimagingAnalysisSoftware_softwareVersion"] = (
                self.software.version
            )
            self.info["Data_grandMeanScaling"] = self.model_fittings[0].data.grand_mean_sc
            self.info["Data_targetIntensity"] = self.model_fittings[
                0
            ].data.target_intensity
            self.info["DesignMatrix_atLocation"] = self.model_fittings[
                0
            ].design_matrix.csv_file
            self.info["DesignMatrix_regressorNames"] = self.model_fittings[
                0
            ].design_matrix.regressors
            self.info["ErrorModel_hasErrorDistribution"] = str(
                self.model_fittings[0].error_model.error_distribution
            )
            self.info["ErrorModel_errorVarianceHomogeneous"] = self.model_fittings[
                0
            ].error_model.variance_homo
            # TODO: replace IRIs by preferred prefixes for readability
            self.info["ErrorModel_varianceMapWiseDependence"] = str(
                self.model_fittings[0].error_model.variance_spatial
            )
            self.info["ErrorModel_hasErrorDependence"] = str(
                self.model_fittings[0].error_model.dependence
            )
            self.info["ErrorModel_dependenceMapWiseDependence"] = str(
                self.model_fittings[0].error_model.dependance_spatial
            )
            self.info["ModelParameterEstimation_withEstimationMethod"] = str(
                self.model_fittings[0].activity.estimation_method
            )
            self.info["ResidualMeanSquaresMap_atLocation"] = self.model_fittings[
                0
            ].rms_map.file.filename
            self.info["ResidualMeanSquaresMap_inWorldCoordinateSystem"] = str(
                self.model_fittings[0].rms_map.coord_space.coordinate_system
            )
            gm_map = self.model_fittings[0].grand_mean_map
            self.info["GrandMeanMap_atLocation"] = gm_map.file.filename
            self.info["GrandMeanMap_inWorldCoordinateSystem"] = str(
                gm_map.coord_space.coordinate_system
            )
            self.info["MaskMap_atLocation"] = self.model_fittings[
                0
            ].mask_map.file.filename
            self.info["MaskMap_inWorldCoordinateSystem"] = str(
                self.model_fittings[0].mask_map.coord_space.coordinate_system
            )

            self.info["ParameterEstimateMaps"] = list()
            # TODO the order of the pe maps matters!!
            for pe in self.model_fittings[0].param_estimates:
                self.info["ParameterEstimateMaps"].append(pe.file.filename)

            self.info["Contrasts"] = list()
            for contrasts in self.contrasts.values():
                for contrast in contrasts:
                    self.info["Contrasts"].append(collections.OrderedDict())
                    self.info["Contrasts"][-1][
                        "StatisticMap_contrastName"
                    ] = contrast.stat_map.contrast_name
                    self.info["Contrasts"][-1][
                        "ContrastWeightMatrix_value"
                    ] = contrast.weights.contrast_weights
                    self.info["Contrasts"][-1]["StatisticMap_statisticType"] = str(
                        contrast.stat_map.stat_type
                    )
                    dof = "StatisticMap_errorDegreesOfFreedom"
                    self.info["Contrasts"][-1][dof] = contrast.stat_map.dof
                    if contrast.stat_map.effdof:
                        edof = "StatisticMap_effectDegreesOfFreedom"
                        self.info["Contrasts"][-1][edof] = contrast.stat_map.effdof
                    self.info["Contrasts"][-1][
                        "StatisticMap_atLocation"
                    ] = contrast.stat_map.file.filename
                    st_world = "StatisticMap_inWorldCoordinateSystem"
                    self.info["Contrasts"][-1][st_world] = str(
                        contrast.stat_map.coord_space.coordinate_system
                    )

                    if contrast.z_stat_map is not None:
                        self.info["Contrasts"][-1][
                            "ZStatisticMap_atLocation"
                        ] = contrast.z_stat_map.file.filename
                        zst_world = "ZStatisticMap_inWorldCoordinateSystem"
                        self.info["Contrasts"][-1][zst_world] = str(
                            contrast.z_stat_map.coord_space.coordinate_system
                        )

                    if contrast.contrast_map:
                        self.info["Contrasts"][-1][
                            "ContrastMap_atLocation"
                        ] = contrast.contrast_map.file.filename
                        c_world = "ContrastMap_inWorldCoordinateSystem"
                        self.info["Contrasts"][-1][c_world] = str(
                            contrast.contrast_map.coord_space.coordinate_system
                        )
                        # TODO: deal when this is not created yet...
                        stderr_loc = "ContrastStandardErrorMap_atLocation"
                        stderr_sys = "ContrastStandardErrorMap_inWorldCoordinateSystem"
                        sderr_explmeansq_map = contrast.stderr_or_expl_mean_sq_map
                        self.info["Contrasts"][-1][
                            stderr_loc
                        ] = sderr_explmeansq_map.file.filename
                        self.info["Contrasts"][-1][stderr_sys] = str(
                            sderr_explmeansq_map.coord_space.coordinate_system
                        )

            self.info["Inferences"] = list()
            for con_est_id, inferences in self.inferences.items():
                for inference in inferences:
                    clustdef = "ClusterDefinitionCriteria_hasConnectivityCriterion"
                    peakdef_mindist = "PeakDefinitionCriteria_minDistanceBetweenPeaks"
                    peakdef_maxpeak = "PeakDefinitionCriteria_maxNumberOfPeaksPerCluster"
                    if clustdef not in self.info:
                        # Assume that all inference have the same cluster def
                        # and peak def > should be stated explicitly in JSON
                        # spec and tested
                        self.info[clustdef] = str(inference.cluster_criteria.connectivity)
                        self.info[peakdef_mindist] = inference.peak_criteria.peak_dist
                        self.info[peakdef_maxpeak] = inference.peak_criteria.num_peak
                    else:
                        if not str(inference.cluster_criteria.connectivity) == str(
                            self.info[clustdef]
                        ):
                            raise Exception(
                                "Inferences using multiple connectivity"
                                + " criteria "
                                + str(inference.cluster_criteria.connectivity)
                                + str(self.info[clustdef])
                                + " not handled yet."
                            )
                        if not (
                            inference.peak_criteria.peak_dist
                            == self.info[peakdef_mindist]
                        ):
                            raise Exception(
                                "Inferences using multiple distance between "
                                + "peaks criteria "
                                + str(inference.peak_criteria.peak_dist)
                                + str(self.info[peakdef_mindist])
                                + " not handled yet."
                            )
                        if (
                            not inference.peak_criteria.num_peak
                            == self.info[peakdef_maxpeak]
                        ):
                            raise Exception(
                                "Inferences using multiple number of peak "
                                + "per cluster criteria "
                                + str(inference.peak_criteria.num_peak)
                                + str(self.info[peakdef_maxpeak])
                                + " not handled yet."
                            )

                    contrast = self._get_contrast(con_est_id)
                    # (model_fitting, pe_map_ids) =
                    # self._get_model_fitting(con_est_id)
                    self.info["Inferences"].append(collections.OrderedDict())
                    self.info["Inferences"][-1][
                        "StatisticMap_contrastName"
                    ] = contrast.stat_map.contrast_name
                    self.info["Inferences"][-1]["HeightThreshold_type"] = str(
                        inference.height_thresh.threshold_type
                    )
                    self.info["Inferences"][-1][
                        "HeightThreshold_value"
                    ] = inference.height_thresh.value
                    self.info["Inferences"][-1]["ExtentThreshold_type"] = str(
                        inference.extent_thresh.threshold_type
                    )
                    clustsize_vox = "ExtentThreshold_clusterSizeInVoxels"
                    self.info["Inferences"][-1][
                        clustsize_vox
                    ] = inference.extent_thresh.extent
                    althyp = "Inference_hasAlternativeHypothesis"
                    self.info["Inferences"][-1][althyp] = str(
                        inference.inference_act.tail
                    )
                    search_loc = "SearchSpaceMaskMap_atLocation"
                    self.info["Inferences"][-1][
                        search_loc
                    ] = inference.search_space.file.filename
                    search_sys = "SearchSpaceMaskMap_inWorldCoordinateSystem"
                    self.info["Inferences"][-1][search_sys] = str(
                        inference.search_space.coord_space.coordinate_system
                    )
                    search_vol_vox = "SearchSpaceMaskMap_searchVolumeInVoxels"
                    self.info["Inferences"][-1][
                        search_vol_vox
                    ] = inference.search_space.search_volume_in_voxels
                    search_vol_units = "SearchSpaceMaskMap_searchVolumeInUnits"
                    self.info["Inferences"][-1][
                        search_vol_units
                    ] = inference.search_space.search_volume_in_units
                    exc_loc = "ExcursionSetMap_atLocation"
                    self.info["Inferences"][-1][
                        exc_loc
                    ] = inference.excursion_set.file.filename
                    exc_sys = "ExcursionSetMap_inWorldCoordinateSystem"
                    self.info["Inferences"][-1][exc_sys] = str(
                        inference.excursion_set.coord_space.coordinate_system
                    )
                    self.info["Inferences"][-1]["Clusters"] = list()
                    clus = "Clusters"
                    pk = "Peaks"
                    cl_size_vox = "SupraThresholdCluster_clusterSizeInVoxels"
                    cl_punc = "SupraThresholdCluster_pValueUncorrected"
                    cl_pfwe = "SupraThresholdCluster_pValueFWER"
                    cl_pfdr = "SupraThresholdCluster_qValueFDR"
                    for cluster in inference.clusters:
                        self.info["Inferences"][-1][clus].append(
                            collections.OrderedDict()
                        )
                        self.info["Inferences"][-1][clus][-1][cl_size_vox] = cluster.size
                        if cluster.punc is not None:
                            self.info["Inferences"][-1][clus][-1][cl_punc] = cluster.punc
                        if cluster.pFWER is not None:
                            self.info["Inferences"][-1][clus][-1][cl_pfwe] = cluster.pFWER
                        if cluster.pFDR is not None:
                            self.info["Inferences"][-1][clus][-1][cl_pfdr] = cluster.pFDR
                        self.info["Inferences"][-1][clus][-1][pk] = list()
                        peaks = list()
                        for peak in cluster.peaks:
                            peaks.append(collections.OrderedDict())
                            if peak.value is not None:
                                peaks[-1]["Peak_value"] = peak.value
                            peaks[-1][
                                "Coordinate_coordinateVector"
                            ] = peak.coordinate.coord_vector_std
                            if peak.p_unc is not None:
                                peaks[-1]["Peak_pValueUncorrected"] = peak.p_unc
                            if peak.p_fwer is not None:
                                peaks[-1]["Peak_pValueFWER"] = peak.p_fwer
                            if peak.p_fdr is not None:
                                peaks[-1]["Peak_qValueFDR"] = peak.p_fdr
                            if peak.equiv_z is not None:
                                peaks[-1]["Peak_equivalentZStatistic"] = peak.equiv_z
                            self.info["Inferences"][-1][clus][-1][pk].append(peaks)

        # TODO: if all world coord system are identical then use
        # self.info['CoordinateSpace_inWorldCoordinateSystem'] = ''
        # self.info['CoordinateSpace_voxelUnits'] = ''

        return self.info

    def get_metadata(self):
        return self.objects

    def _get_model_fitting(self, con_est_id):
        """Retrieve model fitting that corresponds \
           to contrast with identifier 'con_id' \
           from the list of model fitting objects \
           stored in self.model_fittings."""
        for (mpe_id, pe_ids), contrasts in self.contrasts.items():
            for contrast in contrasts:
                if contrast.estimation.id == con_est_id:
                    model_fitting_id = mpe_id
                    pe_map_ids = pe_ids
                    break

        for model_fitting in self.model_fittings:
            if model_fitting.activity.id == model_fitting_id:
                return (model_fitting, pe_map_ids)

        raise Exception("Model fitting of contrast : " + str(con_est_id) + " not found.")

    def _get_contrast(self, con_id):
        """Retrieve contrast with identifier 'con_id' from the list of contrast \
           objects stored in self.contrasts."""
        for contrasts in list(self.contrasts.values()):
            for contrast in contrasts:
                if contrast.estimation.id == con_id:
                    return contrast
        raise Exception("Contrast activity with id: " + str(con_id) + " not found.")

    def parse(self, rdf_data, fmt="turtle"):
        g = rdflib.Graph()
        try:
            g.parse(data=rdf_data, format=fmt)
        except BadSyntax:
            raise self.ParseException("RDFLib was unable to parse the RDF file.")
        return g

    def get_object(self, klass, oid=None, err_if_none=True, **kwargs):

        if oid is not None:
            if oid in self.objects:
                return self.objects[oid]

        query = klass.get_query(oid)
        sd = self.graph.query(query)

        objects = dict()
        if sd:
            for row in sd:
                argums = row.asdict()

                # Convert from rdflib Literal to appropriate Python datatype
                argums = {k: v.toPython() for k, v in argums.items()}
                # Convert URIs to qnames
                argums = {
                    k: (
                        namespace_manager.valid_qualified_name(v)
                        if namespace_manager.valid_qualified_name(v) is not None
                        else v
                    )
                    for k, v in argums.items()
                }

                # for k,v in argums.items():
                #     if not isinstance(drift_type, QualifiedName):
                #         drift_type = namespace_manager.valid_qualified_name(
                # drift_type)

                # Combine with passed arguments
                argums.update(kwargs)
                objects[oid] = klass(oid=oid, **argums)

        self.objects.update(objects)
        if oid is not None:
            if oid not in objects:
                to_return = None
            else:
                to_return = objects[oid]
        else:
            to_return = objects

        if err_if_none and (to_return is None):
            raise Exception("No results found for query:" + query)

        return to_return

    def load_software(self):
        query = """
prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_000005\
6>

SELECT DISTINCT * WHERE {

    ?ni_software_id a prov:SoftwareAgent .

    ?mpe_id a nidm_ModelParameterEstimation: ;
        prov:wasAssociatedWith ?ni_software_id .
}
        """
        sd = self.graph.query(query)

        software = None
        if sd:
            for row in sd:
                args = row.asdict()
                software = self.get_object(NeuroimagingSoftware, args["ni_software_id"])

        if software is None:
            raise Exception("No results found for query:" + query)

        return software

    def load_bundle_export(self):
        # query =  """
        # prefix nidm_softwareVersion: <http://purl.org/nidash/nidm#NIDM_00001\
        # 22>
        # prefix nidm_NIDMResultsExport: <http://purl.org/nidash/nidm#NIDM_000\
        # 0166>

        # SELECT DISTINCT * WHERE
        #     {
        #         ?bundle_id a prov:Bundle .

        #         ?exporter_id a prov:SoftwareAgent .

        #         ?export_id  a nidm_NIDMResultsExport: ;
        #             prov:wasAssociatedWith ?exporter_id .

        #     }
        # """

        query = """
prefix nidm_softwareVersion: <http://purl.org/nidash/nidm#NIDM_0000122>
prefix nidm_NIDMResultsExport: <http://purl.org/nidash/nidm#NIDM_0000166>

SELECT DISTINCT * WHERE
    {
        ?bundle_id a prov:Bundle ;
            prov:qualifiedGeneration ?blank_node .

        ?blank_node a prov:Generation ;
            prov:activity ?export_id ;
            prov:atTime ?export_time .

        ?exporter_id a prov:SoftwareAgent .

        ?export_id a nidm_NIDMResultsExport: ;
            prov:wasAssociatedWith ?exporter_id .

    }
        """

        sd = self.graph.query(query)

        # if len(sd) > 1:
        #     raise Exception('More than one result found for query:' + query)

        exporter = None
        export = None
        bundle = None
        if sd:
            for row in sd:
                args = row.asdict()
                exporter = self.get_object(ExporterSoftware, args["exporter_id"])
                export = self.get_object(NIDMResultsExport, args["export_id"])
                bundle = self.get_object(NIDMResultsBundle, args["bundle_id"])
                export_time = args["export_time"].toPython()
        else:
            raise Exception("No results found for query:" + query)

        return (bundle, exporter, export, export_time)

    def load_modelfitting(self):
        query = """
prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
prefix nidm_hasDriftModel: <http://purl.org/nidash/nidm#NIDM_0000088>
prefix nidm_Data: <http://purl.org/nidash/nidm#NIDM_0000169>
prefix nidm_ErrorModel: <http://purl.org/nidash/nidm#NIDM_0000023>
prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_000005\
6>
prefix nidm_ResidualMeanSquaresMap: <http://purl.org/nidash/nidm#NIDM_0000066>
prefix nidm_MaskMap: <http://purl.org/nidash/nidm#NIDM_0000054>
prefix nidm_GrandMeanMap: <http://purl.org/nidash/nidm#NIDM_0000033>
prefix nlx_Imaginginstrument: <http://uri.neuinfo.org/nif/nifstd/birnlex_2094>
prefix nlx_MagneticResonanceImagingScanner: <http://uri.neuinfo.org/nif/nifstd\
/birnlex_2100>
prefix nlx_PositronEmissionTomographyScanner: <http://uri.neuinfo.org/nif/nifs\
td/ixl_0050000>
prefix nlx_SinglePhotonEmissionComputedTomographyScanner: <http://uri.neuinfo.\
org/nif/nifstd/ixl_0050001>
prefix nlx_MagnetoencephalographyMachine: <http://uri.neuinfo.org/nif/nifstd/i\
xl_0050002>
prefix nlx_ElectroencephalographyMachine: <http://uri.neuinfo.org/nif/nifstd/i\
xl_0050003>
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

    ?mask_id a nidm_MaskMap: ;
        nidm_inCoordinateSpace: ?mask_coordspace_id ;
        prov:wasGeneratedBy ?mpe_id .

    ?gm_id a nidm_GrandMeanMap: ;
        nidm_inCoordinateSpace: ?gm_coordspace_id ;
        prov:wasGeneratedBy ?mpe_id .

    OPTIONAL {
        ?rpv_id a nidm_ReselsPerVoxelMap: ;
            nidm_inCoordinateSpace: ?rpv_coordspace_id ;
            prov:wasGeneratedBy ?mpe_id .
    }
}
        """
        sd = self.graph.query(query)

        model_fittings = list()
        if sd:
            for row in sd:
                row_num = 0
                args = row.asdict()

                # TODO: should software_id really be an input?
                activity = self.get_object(ModelParametersEstimation, args["mpe_id"])

                # Find list of HRF basis
                query_hrf_bases = (
                    """
prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
prefix nidm_hasHRFBasis: <http://purl.org/nidash/nidm#NIDM_0000102>

SELECT DISTINCT * WHERE {
    <"""
                    + args["design_id"]
                    + """> nidm_hasHRFBasis: ?hrf_basis .
}
                """
                )
                hrf_models = None
                sd_hrf = self.graph.query(query_hrf_bases)
                if sd_hrf:
                    hrf_models = list()
                    # TODO: we probably can avoid the loop below
                    for row_hrf in sd_hrf:
                        args_hrf = row_hrf.asdict()
                        hrf_models.append(
                            namespace_manager.valid_qualified_name(args_hrf["hrf_basis"])
                        )

                if "png_id" in args:
                    design_matrix_png = self.get_object(Image, args["png_id"])
                else:
                    design_matrix_png = None

                if "drift_model_id" in args:
                    drift_model = self.get_object(DriftModel, args["drift_model_id"])
                else:
                    drift_model = None

                design_matrix = self.get_object(
                    DesignMatrix,
                    args["design_id"],
                    matrix=None,
                    image_file=design_matrix_png,
                    drift_model=drift_model,
                    hrf_models=hrf_models,
                )
                data = self.get_object(Data, args["data_id"])
                error_model = self.get_object(ErrorModel, args["error_id"])

                # Find list of model parameter estimate maps
                query_pe_maps = (
                    """
prefix nidm_ParameterEstimateMap: <http://purl.org/nidash/nidm#NIDM_0000061>
prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>

SELECT DISTINCT * WHERE {
    ?pe_id a nidm_ParameterEstimateMap: ;
    nidm_inCoordinateSpace: ?pe_coordspace_id ;
    prov:wasGeneratedBy <"""
                    + str(args["mpe_id"])
                    + """> .
}
                """
                )
                param_estimates = list()
                sd_pe_maps = self.graph.query(query_pe_maps)
                if sd_pe_maps:
                    for row_pe in sd_pe_maps:
                        args_pe = row_pe.asdict()
                        pe_map_coordspace = self.get_object(
                            CoordinateSpace, args_pe["pe_coordspace_id"]
                        )

                        param_estimates.append(
                            self.get_object(
                                ParameterEstimateMap,
                                args_pe["pe_id"],
                                coord_space=pe_map_coordspace,
                                pe_num=None,
                            )
                        )

                rms_coord_space = self.get_object(
                    CoordinateSpace, args["rms_coordspace_id"]
                )
                rms_map = self.get_object(
                    ResidualMeanSquares, args["rms_id"], coord_space=rms_coord_space
                )

                mask_coord_space = self.get_object(
                    CoordinateSpace, args["mask_coordspace_id"]
                )
                mask_map = self.get_object(
                    MaskMap, args["mask_id"], coord_space=mask_coord_space
                )

                gm_coord_space = self.get_object(
                    CoordinateSpace, args["gm_coordspace_id"]
                )
                grand_mean_map = self.get_object(
                    GrandMeanMap,
                    args["gm_id"],
                    coord_space=mask_coord_space,
                    mask_file=None,
                )

                if "rpv_coordspace_id" in args:
                    rpv_coord_space = self.get_object(
                        CoordinateSpace, args["rpv_coordspace_id"]
                    )
                    rpv_map = self.get_object(
                        ReselsPerVoxelMap, args["rpv_id"], coord_space=mask_coord_space
                    )
                else:
                    rpv_map = None

                machine = self.get_object(ImagingInstrument, args["machine_id"])

                # Find subject or group(s)
                query_subjects = (
                    """
prefix nidm_Data: <http://purl.org/nidash/nidm#NIDM_0000169>
prefix obo_studygrouppopulation: <http://purl.obolibrary.org/obo/STATO_0000193>

SELECT DISTINCT * WHERE {
    <"""
                    + str(args["data_id"])
                    + """> a nidm_Data: ;
        prov:wasAttributedTo ?person_or_group_id .

    {?person_or_group_id a prov:Person .} UNION
    {?person_or_group_id a obo_studygrouppopulation: .} .

}
                """
                )

                sd_subjects = self.graph.query(query_subjects)
                subjects = None

                if sd_subjects:
                    subjects = list()
                    for row_sub in sd_subjects:
                        args_sub = row_sub.asdict()
                        group = self.get_object(
                            Group, args_sub["person_or_group_id"], err_if_none=False
                        )

                        if group is None:
                            # Try loading as a single subject
                            subject = self.get_object(
                                Person, args_sub["person_or_group_id"]
                            )
                            subjects.append(subject)
                        else:
                            subjects.append(group)

                model_fittings.append(
                    ModelFitting(
                        activity,
                        design_matrix,
                        data,
                        error_model,
                        param_estimates,
                        rms_map,
                        mask_map,
                        grand_mean_map,
                        machine,
                        subjects,
                        rpv_map,
                    )
                )

                con_num = row_num + 1
        else:
            raise Exception("No model fitting found")

        return model_fittings

    def load_contrasts(self, workaround=False):
        if workaround:
            warnings.warn(
                "Using workaround: links between contrast weights"
                + "and contrast estimations are not assessed"
            )
            con_est_att = "."
        else:
            con_est_att = """;
                prov:used ?conw_id ;
                prov:used ?design_id ."""

        query = (
            """
prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_000005\
6>
prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>
prefix obo_contrastweightmatrix: <http://purl.obolibrary.org/obo/STATO_0000323>
prefix nidm_ContrastEstimation: <http://purl.org/nidash/nidm#NIDM_0000001>
prefix nidm_ContrastMap: <http://purl.org/nidash/nidm#NIDM_0000002>
prefix nidm_ContrastStandardErrorMap: <http://purl.org/nidash/nidm#NIDM_000001\
3>
prefix nidm_ContrastExplainedMeanSquareMap: <http://purl.org/nidash/nidm#NIDM_\
0000163>
prefix nidm_StatisticMap: <http://purl.org/nidash/nidm#NIDM_0000076>
prefix nidm_Inference: <http://purl.org/nidash/nidm#NIDM_0000049>
prefix nidm_ConjunctionInference: <http://purl.org/nidash/nidm#NIDM_0000011>
prefix spm_PartialConjunctionInference: <http://purl.org/nidash/spm#SPM_000000\
5>
prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>

SELECT DISTINCT * WHERE {

    ?design_id a nidm_DesignMatrix: .

    ?conw_id a obo_contrastweightmatrix: .

    ?conest_id a nidm_ContrastEstimation: """
            + con_est_att
            + """

    ?mpe_id a nidm_ModelParameterEstimation: ;
        prov:used ?design_id .

    {
    ?conm_id a nidm_ContrastMap: ;
        nidm_inCoordinateSpace: ?conm_coordspace_id ;
        prov:wasGeneratedBy ?conest_id .

    ?constdm_id a nidm_ContrastStandardErrorMap: .
    } UNION
    {
    ?constdm_id a nidm_ContrastExplainedMeanSquareMap: .
    } .

    ?constdm_id nidm_inCoordinateSpace: ?constdm_coordspace_id ;
        prov:wasGeneratedBy ?conest_id .

    ?statm_id a nidm_StatisticMap: ;
        prov:wasGeneratedBy ?conest_id ;
        nidm_contrastName: ?contrast_name ;
        nidm_inCoordinateSpace: ?statm_coordspace_id .

    {
    ?inf_id a nidm_Inference: .
    } UNION {
    ?inf_id a nidm_ConjunctionInference: .
    } UNION {
    ?inf_id a spm_PartialConjunctionInference: .
    } .

    ?inf_id prov:used ?statm_id .

    OPTIONAL {
        ?otherstatm_id a nidm_StatisticMap: ;
        prov:wasGeneratedBy ?conest_id ;
        nidm_inCoordinateSpace: ?otherstatm_coordspace_id .
        FILTER (?otherstatm_id != ?statm_id)
    } .

}
        """
        )

        sd = self.graph.query(query)

        contrasts = dict()
        if sd:
            con_num = 0
            for row in sd:
                args = row.asdict()
                contrast_num = str(con_num)
                contrast_name = args["contrast_name"]

                weights = self.get_object(
                    ContrastWeights, args["conw_id"], contrast_num=contrast_num
                )
                estimation = self.get_object(
                    ContrastEstimation, args["conest_id"], contrast_num=contrast_num
                )

                contraststd_map_coordspace = self.get_object(
                    CoordinateSpace, args["constdm_coordspace_id"]
                )

                if "conm_id" in args:
                    # T-contrast
                    contrast_map_coordspace = self.get_object(
                        CoordinateSpace, args["conm_coordspace_id"]
                    )
                    contrast_map = self.get_object(
                        ContrastMap,
                        args["conm_id"],
                        coord_space=contrast_map_coordspace,
                        contrast_num=contrast_num,
                    )

                    stderr_or_expl_mean_sq_map = self.get_object(
                        ContrastStdErrMap,
                        args["constdm_id"],
                        coord_space=contraststd_map_coordspace,
                        contrast_num=contrast_num,
                        is_variance=False,
                        var_coord_space=None,
                        filepath=None,
                    )
                else:
                    # F-contrast
                    contrast_map = None
                    stderr_or_expl_mean_sq_map = self.get_object(
                        ContrastExplainedMeanSquareMap,
                        args["constdm_id"],
                        coord_space=contraststd_map_coordspace,
                        contrast_num=contrast_num,
                        stat_file=None,
                        sigma_sq_file=None,
                    )

                stat_map_coordspace = self.get_object(
                    CoordinateSpace, args["statm_coordspace_id"]
                )
                stat_map = self.get_object(
                    StatisticMap, args["statm_id"], coord_space=stat_map_coordspace
                )

                zstat_exist = False
                if "otherstatm_id" in args:
                    zstat_exist = True

                    otherstat_map_coordspace = self.get_object(
                        CoordinateSpace, args["otherstatm_coordspace_id"]
                    )
                    otherstat_map = self.get_object(
                        StatisticMap,
                        args["otherstatm_id"],
                        coord_space=stat_map_coordspace,
                    )

                if zstat_exist:
                    con = Contrast(
                        contrast_num,
                        args["contrast_name"],
                        weights,
                        estimation,
                        contrast_map,
                        stderr_or_expl_mean_sq_map,
                        stat_map=otherstat_map,
                        z_stat_map=stat_map,
                    )
                else:
                    con = Contrast(
                        contrast_num,
                        args["contrast_name"],
                        weights,
                        estimation,
                        contrast_map,
                        stderr_or_expl_mean_sq_map,
                        stat_map=stat_map,
                    )

                # Find list of model parameter estimate maps
                query_pe_maps = (
                    """
prefix nidm_ParameterEstimateMap: <http://purl.org/nidash/nidm#NIDM_0000061>

SELECT DISTINCT * WHERE {
    <"""
                    + str(args["conest_id"])
                    + """> prov:used  ?pe_id .
    ?pe_id a nidm_ParameterEstimateMap: .
}
                """
                )
                pe_ids = ()
                sd_pe_maps = self.graph.query(query_pe_maps)
                if sd_pe_maps:
                    for row_pe in sd_pe_maps:
                        args_pe = row_pe.asdict()
                        pe_ids = pe_ids + (NIIRI.qname(str(args_pe["pe_id"])),)

                mpe_id = NIIRI.qname(str(args["mpe_id"]))
                if not (mpe_id, pe_ids) in contrasts:
                    contrasts[(mpe_id, pe_ids)] = [con]
                else:
                    contrasts[(mpe_id, pe_ids)].append(con)

                con_num = con_num + 1

        if not contrasts:
            raise Exception("No contrast found")

        return contrasts

    def load_inferences(self):
        query = """
prefix nidm_ContrastEstimation: <http://purl.org/nidash/nidm#NIDM_0000001>

prefix nidm_Inference: <http://purl.org/nidash/nidm#NIDM_0000049>
prefix nidm_HeightThreshold: <http://purl.org/nidash/nidm#NIDM_0000034>
prefix nidm_ExtentThreshold: <http://purl.org/nidash/nidm#NIDM_0000026>
prefix nidm_PeakDefinitionCriteria: <http://purl.org/nidash/nidm#NIDM_0000063>
prefix nidm_ClusterDefinitionCriteria: <http://purl.org/nidash/nidm#NIDM_00000\
07>
prefix nidm_DisplayMaskMap: <http://purl.org/nidash/nidm#NIDM_0000020>
prefix nidm_ExcursionSetMap: <http://purl.org/nidash/nidm#NIDM_0000025>
prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>
prefix nidm_SearchSpaceMaskMap: <http://purl.org/nidash/nidm#NIDM_0000068>
prefix nidm_ConjunctionInference: <http://purl.org/nidash/nidm#NIDM_0000011>
prefix spm_PartialConjunctionInference: <http://purl.org/nidash/spm#SPM_000000\
5>
prefix nidm_hasClusterLabelsMap: <http://purl.org/nidash/nidm#NIDM_0000098>
prefix nidm_hasMaximumIntensityProjection: <http://purl.org/nidash/nidm#NIDM_0\
000138>

SELECT DISTINCT * WHERE {
    ?con_est_id a nidm_ContrastEstimation: .

    { ?inference_id a nidm_Inference: . }
    UNION { ?inference_id a nidm_ConjunctionInference: . }
    UNION { ?inference_id a spm_PartialConjunctionInference: . } .

    ?inference_id prov:used/prov:wasGeneratedBy ?con_est_id ;
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

    OPTIONAL {?exc_set_id dc:description ?excset_visu_id}
    OPTIONAL {
        ?exc_set_id nidm_hasClusterLabelsMap: ?cluster_map_id .
        ?cluster_map_id nidm_inCoordinateSpace: ?cluster_map_coord_space_id . }
    OPTIONAL {?exc_set_id nidm_hasMaximumIntensityProjection: ?mip_id}

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
                inference = self.get_object(
                    InferenceActivity, args["inference_id"], err_if_none=False
                )

                # Find list of equivalent height thresholds
                query_equiv_threshs = (
                    """
prefix nidm_equivalentThreshold: <http://purl.org/nidash/nidm#NIDM_0000161>

SELECT DISTINCT * WHERE {
    <"""
                    + str(args["height_thresh_id"])
                    + """> nidm_equivalentThreshold: ?equiv_h_thresh_id .
}
                """
                )
                equiv_h_threshs = list()
                sd_equiv_h_threshs = self.graph.query(query_equiv_threshs)
                if sd_equiv_h_threshs:
                    for row_equiv_h in sd_equiv_h_threshs:
                        args_hequiv = row_equiv_h.asdict()

                        equiv_h_threshs.append(
                            self.get_object(
                                HeightThreshold, args_hequiv["equiv_h_thresh_id"]
                            )
                        )

                height_thresh = self.get_object(
                    HeightThreshold,
                    args["height_thresh_id"],
                    equiv_thresh=equiv_h_threshs,
                )

                # Find list of equivalent extent thresholds
                query_equiv_threshs = (
                    """
prefix nidm_equivalentThreshold: <http://purl.org/nidash/nidm#NIDM_0000161>

SELECT DISTINCT * WHERE {
    <"""
                    + str(args["extent_thresh_id"])
                    + """> nidm_equivalentThreshold: ?equiv_e_thresh_id .
}
                """
                )
                equiv_e_threshs = list()
                sd_equiv_e_threshs = self.graph.query(query_equiv_threshs)
                if sd_equiv_e_threshs:
                    for row_equiv_e in sd_equiv_e_threshs:
                        args_eequiv = row_equiv_e.asdict()

                        equiv_e_threshs.append(
                            self.get_object(
                                ExtentThreshold, args_eequiv["equiv_e_thresh_id"]
                            )
                        )

                extent_thresh = self.get_object(
                    ExtentThreshold,
                    args["extent_thresh_id"],
                    equiv_thresh=equiv_e_threshs,
                )
                peak_criteria = self.get_object(
                    PeakCriteria, args["peak_criteria_id"], contrast_num=None
                )
                cluster_criteria = self.get_object(
                    ClusterCriteria, args["cluster_criteria_id"], contrast_num=None
                )

                if "display_mask_id" in args:
                    disp_coordspace = self.get_object(
                        CoordinateSpace, args["disp_coord_space_id"]
                    )
                    # TODO we need to deal with more than 1 DisplayMaskMap
                    disp_mask = [
                        self.get_object(
                            DisplayMaskMap,
                            args["display_mask_id"],
                            contrast_num=None,
                            coord_space=disp_coordspace,
                            mask_num=None,
                        )
                    ]
                else:
                    disp_mask = None

                if "excset_visu_id" in args:
                    excset_visu = self.get_object(Image, args["excset_visu_id"])
                else:
                    excset_visu = None

                if "cluster_map_id" in args:
                    clustermap_coordspace = self.get_object(
                        CoordinateSpace, args["cluster_map_coord_space_id"]
                    )
                    cluster_map = self.get_object(
                        ClusterLabelsMap,
                        args["cluster_map_id"],
                        coord_space=clustermap_coordspace,
                    )
                else:
                    cluster_map = None

                if "mip_id" in args:
                    mip = self.get_object(Image, args["mip_id"])
                else:
                    mip = None

                excset_coordspace = self.get_object(
                    CoordinateSpace, args["excset_coord_space_id"]
                )
                excursion_set = self.get_object(
                    ExcursionSet,
                    args["exc_set_id"],
                    coord_space=excset_coordspace,
                    visu=excset_visu,
                    clust_map=cluster_map,
                    mip=mip,
                )

                searchspace_coordspace = self.get_object(
                    CoordinateSpace, args["search_space_coord_space_id"]
                )
                search_space = self.get_object(
                    SearchSpace,
                    args["search_space_id"],
                    coord_space=searchspace_coordspace,
                )

                # TODO
                software_id = self.software.id

                # Find list of clusters
                query_clusters = (
                    """
prefix nidm_SupraThresholdCluster: <http://purl.org/nidash/nidm#NIDM_0000070>
prefix nidm_ClusterCenterOfGravity: <http://purl.org/nidash/nidm#NIDM_0000140>
prefix nidm_clusterLabelId: <http://purl.org/nidash/nidm#NIDM_0000082>

SELECT DISTINCT * WHERE {
    ?cluster_id a nidm_SupraThresholdCluster: ;
        nidm_clusterLabelId: ?cluster_num ;
        prov:wasDerivedFrom <"""
                    + str(args["exc_set_id"])
                    + """> .

    OPTIONAL {
    ?cog_id a nidm_ClusterCenterOfGravity: ;
        prov:wasDerivedFrom ?cluster_id .
    }
}
                """
                )
                clusters = list()
                sd_clusters = self.graph.query(query_clusters)
                if sd_clusters:
                    for row_cluster in sd_clusters:
                        args_cl = row_cluster.asdict()

                        if "cog_id" in args_cl:
                            cog = self.get_object(
                                CenterOfGravity,
                                args_cl["cog_id"],
                                cluster_num=args_cl["cluster_num"].toPython(),
                            )
                        else:
                            cog = None

                        # Find list of peaks
                        query_peaks = (
                            """
prefix nidm_Peak: <http://purl.org/nidash/nidm#NIDM_0000062>

SELECT DISTINCT * WHERE {
    ?peak_id a nidm_Peak: ;
    prov:wasDerivedFrom <"""
                            + str(args_cl["cluster_id"])
                            + """> .
}
                        """
                        )
                        peaks = list()
                        sd_peaks = self.graph.query(query_peaks)
                        if sd_peaks:
                            for row_peak in sd_peaks:
                                args_peak = row_peak.asdict()

                                peaks.append(self.get_object(Peak, args_peak["peak_id"]))

                        clusters.append(
                            self.get_object(
                                Cluster, args_cl["cluster_id"], peaks=peaks, cog=cog
                            )
                        )

                # Dictionary of (key, value) pairs where key is the identifier
                # of a ContrastEstimation object and value is an object of type
                # Inference describing the inference step in NIDM-Results (main
                # activity: Inference)
                # TODO: if key exist we need to append!
                inferences[NIIRI.qname(args["con_est_id"])] = [
                    Inference(
                        inference,
                        height_thresh,
                        extent_thresh,
                        peak_criteria,
                        cluster_criteria,
                        disp_mask,
                        excursion_set,
                        clusters,
                        search_space,
                        software_id,
                    )
                ]

        return inferences

    def serialize(self, destination, fmt="nidm", overwrite=False, last_used_con_id=0):

        if fmt == "nidm":
            exporter = NIDMExporter(
                version="1.3.0", out_dir=destination.replace(".nidm.zip", "")
            )
            exporter.model_fittings = self.model_fittings
            exporter.contrasts = self.contrasts
            exporter.inferences = self.inferences
            exporter.bundle = self.bundle
            # exporter.exporter = ExporterSoftware(
            #   'nidmresults', nidmresults.__version__)
            exporter.software = self.software
            exporter.prepend_path = self.zip_path
            exporter.exporter = self.exporter
            exporter.bundle_ent = self.bundle
            exporter.export_act = self.export_act
            exporter.export_time = self.export_time
            exporter.export()

        elif fmt == "mkda":
            if not destination.endswith(".csv"):
                destination = destination + ".csv"
            csvfile = destination

            if overwrite:
                add_header = True
                open_mode = "wb"
            else:
                add_header = False
                open_mode = "ab"

            with open(csvfile, open_mode) as fid:
                writer = csv.writer(fid, delimiter="\t")
                if add_header:
                    writer.writerow(
                        ["9", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]
                    )
                    writer.writerow(
                        [
                            "x",
                            "y",
                            "z",
                            "Study",
                            "Contrast",
                            "N",
                            "FixedRandom",
                            "CoordSys",
                            "Name",
                        ]
                    )

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
                            "Unrecognised space for "
                            + str(exc_set.coord_space.coordinate_system)
                        )

                    stat_map = exc_set.inference.stat_map

                    con_name = stat_map.contrast_name.replace(" ", "_").replace(":", "")

                    # FIXME: need to deal with more than one group
                    self.N = (
                        stat_map.contrast_estimation.param_estimate.model_param_estimation.data.group_or_sub.num_subjects  # noqa
                    )

                    if con_name in con_ids:
                        con_id = con_ids[con_name]
                    else:
                        con_id = max(con_ids.values()) + 1
                        con_ids[con_name] = con_id

                    writer.writerow(
                        [
                            peak.coordinate.coord_vector[0],
                            peak.coordinate.coord_vector[1],
                            peak.coordinate.coord_vector[2],
                            self.study_name,
                            con_id,
                            self.N,
                            self.FixedRandom,
                            space,
                            con_name,
                        ]
                    )

                return con_ids
