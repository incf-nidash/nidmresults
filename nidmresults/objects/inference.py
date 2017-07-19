"""
Objects describing the Inference activity, its inputs and outputs as specified
in NIDM-Results.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""
from nidmresults.objects.constants import *
from nidmresults.objects.generic import *
import uuid
from math import erf, sqrt
import rdflib
from prov.model import Literal
from prov.constants import XSD_FLOAT


class Inference(object):

    """
    Object representing an Inference step: including an Inference activity, its
    inputs and outputs.
    """

    def __init__(
            self, inference, height_thresh, extent_thresh,
            peak_criteria, cluster_criteria, disp_mask, excursion_set,
            clusters, search_space, software_id):
        super(Inference, self).__init__()
        self.excursion_set = excursion_set
        self.inference_act = inference
        self.height_thresh = height_thresh
        self.extent_thresh = extent_thresh
        self.clusters = clusters
        self.software_id = software_id
        self.peak_criteria = peak_criteria
        self.cluster_criteria = cluster_criteria
        self.disp_mask = disp_mask
        self.search_space = search_space


class InferenceActivity(NIDMObject):

    """
    Object representing an Inference activity.
    """

    def __init__(self, oid=None, tail=None, label=None, contrast_name=None):
        super(InferenceActivity, self).__init__(oid=oid)
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = NIDM_INFERENCE
        self.prov_type = PROV['Activity']
        if tail is None:
            tail = NIDM_ONE_TAILED_TEST
        self.tail = tail
        if label is None:
            label = "Inference"
            if contrast_name:
                label += ": " + self.contrast_name
        self.label = label


    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_Inference: <http://purl.org/nidash/nidm#NIDM_0000049>
        prefix nidm_hasAlternativeHypothesis: <http://purl.org/nidash/nidm#NIDM_0000097>

            SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_Inference: ;
                rdfs:label ?label ;
                nidm_hasAlternativeHypothesis: ?tail .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """

        # In FSL we have a single thresholding (extent, height) applied to all
        # contrasts
        # FIXME: Deal with two-tailed inference?
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['label'], self.label),
            (NIDM_HAS_ALTERNATIVE_HYPOTHESIS, self.tail)))


class ExcursionSet(NIDMObject):

    """
    Object representing a ExcursionSet entity.
    """

    def __init__(self, location, coord_space, visualisation=None,
                 oid=None, format=None, label=None,
                 sha=None, filename=None, inference=None, suffix='',
                 clust_map=None):
        super(ExcursionSet, self).__init__(oid)
        if not filename:
            filename = 'ExcursionSet' + suffix + '.nii.gz'
        else:
            filename = location
        self.filename = filename
        self.file = NIDMFile(self.id, location, filename, sha)
        self.type = NIDM_EXCURSION_SET_MAP
        self.prov_type = PROV['Entity']
        if visualisation is not None:
            visu_filename = 'ExcursionSet' + suffix + '.png'
            self.visu = Image(visualisation, visu_filename)
        if label is None:
            label = "Excursion Set Map"
        self.label = label
        self.coord_space = coord_space
        self.clust_map = clust_map
        # FIXME Not used for export yet (only for reading)
        self.inference = inference

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ExcursionSetMap: <http://purl.org/nidash/nidm#NIDM_0000025>
        prefix nidm_hasClusterLabelsMap: <http://purl.org/nidash/nidm#NIDM_0000098>
        prefix nidm_hasMaximumIntensityProjection: <http://purl.org/nidash/nidm#NIDM_0000138>
        prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>
        prefix nidm_numberOfSignificantClusters: <http://purl.org/nidash/nidm#NIDM_0000111>
        prefix nidm_pValue: <http://purl.org/nidash/nidm#NIDM_0000114>

        SELECT DISTINCT * WHERE {

        """ + oid_var + """ a nidm_ExcursionSetMap: ;
            prov:atLocation ?location ;
            rdfs:label ?label ;
            dct:format ?format ;
            nfo:fileName ?filename ;
            crypto:sha512 ?sha ;
            OPTIONAL {""" + oid_var + """ dc:description ?visualisation } .
        }
        ORDER BY ?peak_label

        """
        return query


    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        # Create "Excursion set" entity
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], self.label),
            (DC['description'], self.visu.id)
        ))

        if self.clust_map is not None:
            self.add_attributes((
                (NIDM_HAS_CLUSTER_LABELS_MAP, self.clust_map.id),
            ))


class ClusterLabelsMap(NIDMObject):

    """
    Object representing a ClusterLabelsMap entity.
    """

    def __init__(self, location, coord_space,
                 oid=None, format=None, label=None,
                 sha=None, filename=None, suffix='', temporary=False):
        super(ClusterLabelsMap, self).__init__(oid)
        if not filename:
            filename = 'ClusterLabels' + suffix + '.nii.gz'
        self.filename = filename
        self.file = NIDMFile(self.id, location, filename, sha,
                             temporary=temporary)
        self.type = NIDM_CLUSTER_LABELS_MAP
        self.prov_type = PROV['Entity']
        if label is None:
            label = "Cluster Labels Map"
        self.label = label
        self.coord_space = coord_space

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        # Create "Cluster Labels Map" entity
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], self.label)
        ))


class HeightThreshold(NIDMObject):

    """
    Object representing a HeightThreshold entity.
    """

    def __init__(self, stat_threshold=None, p_corr_threshold=None,
                 p_uncorr_threshold=None, threshold_type=None, value=None, label=None,
                 version={'num': '1.3.0'}, oid=None):
        super(HeightThreshold, self).__init__(oid=oid)
        if not stat_threshold and not p_corr_threshold and \
           not p_uncorr_threshold and not value:
            raise Exception('No threshold defined')

        thresh_desc = ""
        if stat_threshold is not None:
            thresh_desc = "Z>" + str(self.stat_threshold)
            if version['num'] == "1.0.0":
                user_threshold_type = "Z-Statistic"
            else:
                threshold_type = OBO_STATISTIC
                value = stat_threshold
        elif p_uncorr_threshold is not None:
            thresh_desc = "p<" + \
                str(p_uncorr_threshold) + " (uncorrected)"
            if version['num'] == "1.0.0":
                user_threshold_type = "p-value uncorrected"
            else:
                threshold_type = NIDM_P_VALUE_UNCORRECTED_CLASS
                value = p_uncorr_threshold
        elif p_corr_threshold is not None:
            thresh_desc = "p<" + str(self.p_corr_threshold) + " (FWE)"
            if version['num'] == "1.0.0":
                user_threshold_type = "p-value FWE"
            else:
                threshold_type = OBO_P_VALUE_FWER
                value = p_corr_threshold

        if version['num'] == "1.0.0":
            self.user_threshold_type = user_threshold_type
            self.p_uncorr_threshold = p_uncorr_threshold
            self.p_corr_threshold = p_corr_threshold
            self.stat_threshold = stat_threshold
        else:
            self.value = value
            self.threshold_type = threshold_type

        if not label:
            self.label = "Height Threshold: " + thresh_desc
        else:
            self.label = label

        self.id = NIIRI[str(uuid.uuid4())]
        self.type = NIDM_HEIGHT_THRESHOLD
        self.prov_type = PROV['Entity']
        

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_HeightThreshold: <http://purl.org/nidash/nidm#NIDM_0000034>
        prefix nidm_hasAlternativeHypothesis: <http://purl.org/nidash/nidm#NIDM_0000097>

            SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_HeightThreshold: ;
                rdfs:label ?label ;
                prov:value ?value .
        }
        """
        return query


    def export(self, version, export_dir):
        """
        Create prov entities and activities.
        """


        atts = [
            (PROV['type'], self.type),
            (PROV['label'], self.label),
        ]

        if version['num'] == "1.0.0":
            atts += [
                (NIDM_USER_SPECIFIED_THRESHOLD_TYPE, self.user_threshold_type),
                (PROV['value'], self.stat_threshold),
                (NIDM_P_VALUE_UNCORRECTED, self.p_uncorr_threshold),
                (NIDM_P_VALUE_FWER, self.p_corr_threshold)
                ]
        else:
            atts += [
                (PROV['type'], threshold_type),
                (PROV['value'], value)
                ]

        self.add_attributes([(k, v) for k, v in atts if v is not None])


class ExtentThreshold(NIDMObject):

    """
    Object representing an ExtentThreshold entity.
    """

    def __init__(self, extent=None, p_corr=None, p_uncorr=None, extent_rsl=None, 
            label=None, version={'num': '1.3.0'}, value=None, oid=None):
        super(ExtentThreshold, self).__init__(oid=oid)
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = NIDM_EXTENT_THRESHOLD
        self.prov_type = PROV['Entity']

        thresh_desc = ""
        if extent is not None:
            thresh_desc = "k>" + str(extent)
            # NIDM-Results 1.0.0
            user_threshold_type = "Cluster-size in voxels"
            # NIDM-Results > 1.0.0
            threshold_type = OBO_STATISTIC
        elif not p_uncorr is None:
            thresh_desc = "p<" + str(self.p_uncorr) + " (uncorrected)"
            # NIDM-Results 1.0.0
            user_threshold_type = "p-value uncorrected"
            # NIDM-Results > 1.0.0
            threshold_type = NIDM_P_VALUE_UNCORRECTED_CLASS
            value = p_uncorr
        elif not p_corr is None:
            thresh_desc = "p<" + str(p_corr) + " (FWE)"
            # NIDM-Results 1.0.0
            user_threshold_type = "p-value FWE"
            # NIDM-Results > 1.0.0
            threshold_type = OBO_P_VALUE_FWER
            value = p_corr
        else:
            thresh_desc = "k>=0"
            extent = 0
            if version['num'] == "1.0.0":
                p_uncorr = 1.0
                p_corr = 1.0
                user_threshold_type = None
            else:
                threshold_type = OBO_STATISTIC

        if version['num'] == "1.0.0":
            self.user_threshold_type = user_threshold_type
            self.p_uncorr = p_uncorr
            self.p_corr = p_corr

        self.extent = extent
        self.extent_rsl = extent_rsl

        if label is None:
            self.label = "Extent Threshold: " + thresh_desc
        else:
            self.label = label


    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ExtentThreshold: <http://purl.org/nidash/nidm#NIDM_0000026>
        prefix nidm_clusterSizeInVoxels: <http://purl.org/nidash/nidm#NIDM_0000084>
        prefix nidm_clusterSizeInResels: <http://purl.org/nidash/nidm#NIDM_0000156>

            SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ExtentThreshold: ;
                rdfs:label ?label  .
            OPTIONAL {""" + oid_var + """ prov:value ?value .} .
            OPTIONAL {""" + oid_var + """ nidm_clusterSizeInVoxels: ?extent .} .
            OPTIONAL {""" + oid_var + """ nidm_clusterSizeInResels: ?extent_rsl .} .
        }
        """
        return query

    def export(self, version, export_dir):
        """
        Create prov entities and activities.
        """
        atts = [
            (PROV['type'], self.type),
        ]

        atts += [
            (PROV['label'], self.label)
        ]

        if self.extent_rsl is not None:
            atts += [
                (NIDM_CLUSTER_SIZE_IN_RESELS, self.extent_rsl),
            ]

        if self.extent is not None:
            atts += [
                (NIDM_CLUSTER_SIZE_IN_VOXELS, self.extent),
            ]

        if version['num'] == "1.0.0":
            atts += [
                (NIDM_USER_SPECIFIED_THRESHOLD_TYPE, self.user_threshold_type),
                (NIDM_P_VALUE_UNCORRECTED, self.p_uncorr),
                (NIDM_P_VALUE_FWER, self.p_corr)
            ]
        else:
            atts += [
                (PROV['type'], threshold_type)
            ]
            if self.extent is None:
                atts += [
                    (PROV['value'], value)
                ]

        self.add_attributes([(k, v) for k, v in atts if v is not None])


class Cluster(NIDMObject):

    """
    Object representing a Cluster entity.
    """

    def __init__(self, cluster_num, size, pFWER, peaks,
                 x=None, y=None, z=None, x_std=None, y_std=None, z_std=None,
                 suffix='', clust_size_resels=None, pFDR=None, punc=None, 
                 label=None, oid=None):
        super(Cluster, self).__init__(oid=oid)
        self.num = cluster_num
        self.id = NIIRI[str(uuid.uuid4())]
        if x and y and z:
            self.cog = CenterOfGravity(
                cluster_num, x=x, y=y, z=z, x_std=x_std, y_std=y_std, z_std=z_std)
        else:
            self.cog = None
        self.peaks = peaks
        self.size = size
        self.pFWER = pFWER
        self.type = NIDM_SIGNIFICANT_CLUSTER
        self.prov_type = PROV['Entity']
        if not label:
            cluster_naming = "Supra-Threshold Cluster"
            self.label = "%s %04d" % (cluster_naming, self.num)
        else:
            self.label = label


    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_SupraThresholdCluster: <http://purl.org/nidash/nidm#NIDM_0000070>
        prefix nidm_clusterSizeInVoxels: <http://purl.org/nidash/nidm#NIDM_0000084>
        prefix nidm_clusterLabelId: <http://purl.org/nidash/nidm#NIDM_0000082>
        prefix nidm_clusterSizeInResels: <http://purl.org/nidash/nidm#NIDM_0000156>
        prefix nidm_pValueUncorrected: <http://purl.org/nidash/nidm#NIDM_0000116>
        prefix nidm_pValueFWER: <http://purl.org/nidash/nidm#NIDM_0000115>
        prefix nidm_qValueFDR: <http://purl.org/nidash/nidm#NIDM_0000119>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_SupraThresholdCluster: ;
                rdfs:label ?label ;
                nidm_clusterSizeInVoxels: ?size ;
                nidm_clusterLabelId: ?cluster_num .

            OPTIONAL {""" + oid_var + """ nidm_clusterSizeInResels: ?clust_size_resels .} .
            OPTIONAL {""" + oid_var + """ nidm_pValueUncorrected: ?punc .} .
            OPTIONAL {""" + oid_var + """ nidm_pValueFWER: ?pFWER .} .
            OPTIONAL {""" + oid_var + """ nidm_qValueFDR: ?pFDR .} .
        }
        """
        return query


    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        if nidm_version['num'] in ["1.0.0", "1.1.0"]:
            cluster_naming = "Significant Cluster"
        else:
            cluster_naming = "Supra-Threshold Cluster"

        # FIXME deal with multiple contrasts
        self.add_attributes((
            (PROV['type'], NIDM_SIGNIFICANT_CLUSTER),
            (PROV['label'], "%s %04d" % (cluster_naming, self.num)),
            (NIDM_CLUSTER_LABEL_ID, self.num),
            (NIDM_CLUSTER_SIZE_IN_VOXELS, self.size),
            (NIDM_P_VALUE_FWER, Literal(self.pFWER, datatype=XSD_FLOAT))))


class DisplayMaskMap(NIDMObject):

    """
    Object representing a DisplayMaskMap entity.
    """
    def __init__(self, contrast_num, mask_file, mask_num, coord_space,
                 sha=None, filename=None, format=None, label=None, oid=None):
        super(DisplayMaskMap, self).__init__(oid=oid)
        self.id = NIIRI[str(uuid.uuid4())]
        if not filename:
            filename = 'DisplayMask' + str(self.mask_num) + '.nii.gz'
        self.file = NIDMFile(self.id, mask_file, filename,  
            sha=sha, format=format)
        self.coord_space = coord_space
        self.type = NIDM_DISPLAY_MASK_MAP
        self.prov_type = PROV['Entity']
        if not label:
            self.label = "Display Mask Map " + str(mask_num)
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_DisplayMaskMap: <http://purl.org/nidash/nidm#NIDM_0000020>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_DisplayMaskMap: ;
                rdfs:label ?label ;
                nfo:fileName ?filename ;
                crypto:sha512 ?sha ;
                prov:atLocation ?mask_file ;
                dct:format ?format .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['label'], self.label),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id)
            ))


class PeakCriteria(NIDMObject):

    """
    Object representing a PeakCriteria entity.
    """

    def __init__(self, contrast_num, num_peak, peak_dist, label=None, oid=None):
        super(PeakCriteria, self).__init__(oid=oid)
        self.id = NIIRI[str(uuid.uuid4())]
        self.num_peak = num_peak
        self.peak_dist = peak_dist
        self.type = NIDM_PEAK_DEFINITION_CRITERIA
        self.prov_type = PROV['Entity']
        if not label:
            self.label = "Peak Definition Criteria"
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_PeakDefinitionCriteria: <http://purl.org/nidash/nidm#NIDM_0000063>
        prefix nidm_minDistanceBetweenPeaks: <http://purl.org/nidash/nidm#NIDM_0000109>
        prefix nidm_maxNumberOfPeaksPerCluster: <http://purl.org/nidash/nidm#NIDM_0000108>

            SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_PeakDefinitionCriteria: ;
                rdfs:label ?label ;
                nidm_minDistanceBetweenPeaks: ?peak_dist ;
                nidm_maxNumberOfPeaksPerCluster: ?num_peak .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        num_peak = ()
        if self.num_peak:
            num_peak = (NIDM_MAXNUMBEROFPEAKSPERCLUSTER, self.num_peak)

        # Create "Peak definition criteria" entity
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['label'], self.label),
            (NIDM_MIN_DISTANCE_BETWEEN_PEAKS, self.peak_dist)
            ) + num_peak)


class ClusterCriteria(NIDMObject):

    """
    Object representing a ClusterCriteria entity.
    """

    def __init__(self, contrast_num, connectivity, label=None, oid=None):
        super(ClusterCriteria, self).__init__(oid=oid)
        self.id = NIIRI[str(uuid.uuid4())]
        self.connectivity = connectivity
        self.type = NIDM_CLUSTER_DEFINITION_CRITERIA
        self.prov_type = PROV['Entity']
        if not label:
            self.label = "Cluster Connectivity Criterion: " + str(self.connectivity)
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ClusterDefinitionCriteria: <http://purl.org/nidash/nidm#NIDM_0000007>
        prefix nidm_hasConnectivityCriterion: <http://purl.org/nidash/nidm#NIDM_0000099>

            SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ClusterDefinitionCriteria: ;
                rdfs:label ?label ;
                nidm_hasConnectivityCriterion: ?connectivity .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        # Create "Cluster definition criteria" entity
        if isinstance(voxel_conn, int):
            if self.connectivity == 6:
                voxel_conn = NIDM_VOXEL6CONNECTED
            elif self.connectivity == 18:
                voxel_conn = NIDM_VOXEL18CONNECTED
            elif self.connectivity == 26:
                voxel_conn = NIDM_VOXEL26CONNECTED

        # FIXME if connectivity is missing
        if self.connectivity is not None:
            atts = (
                (PROV['type'], self.type),
                (PROV['label'], self.label),
                (NIDM_HAS_CONNECTIVITY_CRITERION, voxel_conn))
        else:
            atts = (
                (PROV['type'], NIDM_CLUSTER_DEFINITION_CRITERIA),
                (PROV['label'], label))

        self.add_attributes(atts)


class CenterOfGravity(NIDMObject):

    """
    Object representing a CenterOfGravity entity.
    """

    def __init__(self, cluster_num, x=None, y=None, z=None, x_std=None,
                 y_std=None, z_std=None, oid=None):
        super(CenterOfGravity, self).__init__(oid=oid)
        self.cluster_num = cluster_num
        self.id = NIIRI[str(uuid.uuid4())]
        self.coordinate = Coordinate("%04d" % cluster_num, x=x, y=y, z=z,
                                     x_std=x_std, y_std=y_std, z_std=z_std)
        self.type = NIDM_CLUSTER_CENTER_OF_GRAVITY
        self.prov_type = PROV['Entity']

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        label = "Center of gravity " + str(self.cluster_num)

        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['label'], label),
            (PROV['location'], self.coordinate.id)))


class SearchSpace(NIDMObject):

    """
    Object representing a SearchSpace entity.
    """

    def __init__(self, search_space_file, vol_in_voxels, vol_in_units,
                 vol_in_resels, resel_size_in_voxels, dlh,
                 random_field_stationarity, noise_fwhm_in_voxels,
                 noise_fwhm_in_units, coord_space, 
                 expected_num_voxels=None, expected_num_clusters=None,
                 height_critical_fwe05=None, height_critical_fdr05=None,
                 extent_critical_fwe05=None, extent_critical_fdr05=None,
                 search_vol_geom=None, filename=None, sha=None, format=None,
                 label=None, oid=None):
        super(SearchSpace, self).__init__(oid=oid)
        self.id = NIIRI[str(uuid.uuid4())]
        if not filename:
            filename = 'SearchSpaceMask.nii.gz'
        self.file = NIDMFile(self.id, search_space_file, filename,
            sha=sha, format=format)
        self.coord_space = coord_space
        self.resel_size_in_voxels = resel_size_in_voxels
        self.dlh = dlh
        self.search_volume_in_voxels = vol_in_voxels
        self.search_volume_in_units = vol_in_units
        self.search_volume_in_resels = vol_in_resels
        self.rf_stationarity = random_field_stationarity
        self.noise_fwhm_in_voxels = noise_fwhm_in_voxels
        self.noise_fwhm_in_units = noise_fwhm_in_units
        self.type = NIDM_SEARCH_SPACE_MASK_MAP
        self.prov_type = PROV['Entity']
        self.label = "Search Space Mask Map"

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_SearchSpaceMaskMap: <http://purl.org/nidash/nidm#NIDM_0000068>
        prefix nidm_expectedNumberOfVoxelsPerCluster: <http://purl.org/nidash/nidm#NIDM_0000143>
        prefix nidm_expectedNumberOfClusters: <http://purl.org/nidash/nidm#NIDM_0000141>
        prefix nidm_heightCriticalThresholdFWE05: <http://purl.org/nidash/nidm#NIDM_0000147>
        prefix nidm_heightCriticalThresholdFDR05: <http://purl.org/nidash/nidm#NIDM_0000146>
        prefix nidm_searchVolumeInVoxels: <http://purl.org/nidash/nidm#NIDM_0000121>
        prefix nidm_searchVolumeInUnits: <http://purl.org/nidash/nidm#NIDM_0000136>
        prefix nidm_searchVolumeInResels: <http://purl.org/nidash/nidm#NIDM_0000149>
        prefix nidm_reselSizeInVoxels: <http://purl.org/nidash/nidm#NIDM_0000148>
        prefix nidm_noiseFWHMInVoxels: <http://purl.org/nidash/nidm#NIDM_0000159>
        prefix nidm_noiseFWHMInUnits: <http://purl.org/nidash/nidm#NIDM_0000157>
        prefix nidm_randomFieldStationarity: <http://purl.org/nidash/nidm#NIDM_0000120>
        prefix spm_smallestSignificantClusterSizeInVoxelsFWE05: <http://purl.org/nidash/spm#SPM_0000014>
        prefix spm_smallestSignificantClusterSizeInVoxelsFDR05: <http://purl.org/nidash/spm#SPM_0000013>
        prefix spm_searchVolumeReselsGeometry: <http://purl.org/nidash/spm#SPM_0000010>
        SELECT DISTINCT * WHERE {

        """ + oid_var + """ a nidm_SearchSpaceMaskMap: ;
            rdfs:label ?label ;
            nidm_searchVolumeInVoxels: ?vol_in_voxels ;
            nidm_searchVolumeInUnits: ?vol_in_units ;
            nidm_searchVolumeInResels: ?vol_in_resels ;
            nidm_reselSizeInVoxels: ?resel_size_in_voxels ;
            nidm_reselSizeInVoxels: ?resel_size_in_voxels ;
            nidm_noiseFWHMInVoxels: ?noise_fwhm_in_voxels ;
            nidm_noiseFWHMInUnits: ?noise_fwhm_in_units ;
            nidm_randomFieldStationarity: ?random_field_stationarity ;
            prov:atLocation ?search_space_file ;
            dct:format ?format ;
            nfo:fileName ?filename ;
            crypto:sha512 ?sha .
        
            OPTIONAL {""" + oid_var + """ nidm_expectedNumberOfVoxelsPerCluster: ?expected_num_voxels } .
            OPTIONAL {""" + oid_var + """ nidm_expectedNumberOfClusters: ?expected_num_clusters } .
            OPTIONAL {""" + oid_var + """ nidm_heightCriticalThresholdFWE05: ?height_critical_fwe05 } .
            OPTIONAL {""" + oid_var + """ nidm_heightCriticalThresholdFDR05: ?height_critical_fdr05 } .
            OPTIONAL {""" + oid_var + """ spm_smallestSignificantClusterSizeInVoxelsFWE05: ?extent_critical_fwe05 } .
            OPTIONAL {""" + oid_var + """ spm_smallestSignificantClusterSizeInVoxelsFDR05: ?extent_critical_fdr05 } .
            OPTIONAL {""" + oid_var + """ spm_smallestSignificantClusterSizeInVoxelsFDR05: ?extent_critical_fdr05 } .
            OPTIONAL {""" + oid_var + """ spm_searchVolumeReselsGeometry: ?search_vol_geom } .
            

        }

        """
        return query

    # Generate prov for search space entity generated by the inference activity
    def export(self, version, export_dir):
        """
        Create prov entities and activities.
        """
        atts = (
            (PROV['label'], self.label),
            (PROV['type'], NIDM_SEARCH_SPACE_MASK_MAP),
            (NIDM_RANDOM_FIELD_STATIONARITY, self.rf_stationarity),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (NIDM_SEARCH_VOLUME_IN_VOXELS, self.search_volume_in_voxels),
            (NIDM_SEARCH_VOLUME_IN_UNITS, self.search_volume_in_units),
            (NIDM_SEARCH_VOLUME_IN_RESELS, self.search_volume_in_resels),
            (NIDM_RESEL_SIZE_IN_VOXELS, self.resel_size_in_voxels),
            (NIDM_NOISE_ROUGHNESS_IN_VOXELS, self.dlh))

        # Noise FWHM was introduced in NIDM-Results 1.1.0
        if self.noise_fwhm_in_voxels is not None:
            if (version['major'] > 1) or \
               (version['major'] >= 1 and
                    (version['minor'] > 0 or version['revision'] > 0)):
                atts = atts + (
                    (NIDM_NOISE_FWHM_IN_VOXELS, self.noise_fwhm_in_voxels),
                    (NIDM_NOISE_FWHM_IN_UNITS, self.noise_fwhm_in_units))

        # Create "Search Space Mask map" entity
        self.add_attributes(atts)


class Coordinate(NIDMObject):

    """
    Object representing a Coordinate entity.
    """

    def __init__(self, label_id, coord_vector=None, coord_vector_std=None,
                 x=None, y=None, z=None, x_std=None, y_std=None, z_std=None,
                 label=None, oid=None):

        super(Coordinate, self).__init__(oid=oid)
        # FIXME: coordiinate_id should not be determined externally
        self.id = NIIRI[str(uuid.uuid4())]
        self.label_id = label_id
        if x is not None and y is not None and z is not None:
            self.coord_vector = [x, y, z]
        else:
            if isinstance(coord_vector, rdflib.term.Literal):
                coord_vector = json.loads(coord_vector)
            self.coord_vector = coord_vector
        if x_std is not None and y_std is not None and z_std is not None:
            self.coord_vector_std = [x_std, y_std, z_std]
        else:
            self.coord_vector_std = coord_vector_std
        self.type = NIDM_COORDINATE
        self.prov_type = PROV['Entity']
        if label is not None:
            self.label = label
        else:
            self.label = "Coordinate " + self.label_id

    def __str__(self):
        return '%s\t%s' % (self.label, self.coord_vector)

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        # We can not have this as a dictionnary because we want to keep the
        # duplicate prov:type attribute
        atts = (  # (PROV['type'],PROV['Location']),
            (PROV['type'], NIDM_COORDINATE),
            (PROV['type'], PROV['Location']),
            (PROV['label'], self.label),
            (NIDM_COORDINATE_VECTOR_IN_VOXELS, json.dumps(self.coord_vector))
            )

        # FSL unnormalised subject-level analyses do not provide coordinates in
        # voxels
        if self.coord_vector_std is not None:
            atts = atts +\
                ((NIDM_COORDINATE_VECTOR, json.dumps(self.coord_vector_std)),)

        self.add_attributes(atts)


class Peak(NIDMObject):

    """
    Object representing a Peak entity.
    """

    def __init__(self, equiv_z, p_unc=None, p_fwer=None, label=None,
                 coord_label=None, exc_set_id=None, oid=None, suffix='',
                 p_fdr=None, value=None, *args, **kwargs):
        super(Peak, self).__init__(oid)
        # FIXME: Currently assumes less than 10 clusters per contrast
        # cluster_num = cluster_index
        # FIXME: Currently assumes less than 100 peaks
        if oid is not None:
            self.label = label
            peak_unique_id = label[5:]
            peak_index = peak_unique_id
            # cluster_index, peak_index = peak_unique_id.split("_")
        else:
            peak_unique_id = suffix
            self.label = "Peak " + peak_unique_id
        self.equiv_z = equiv_z
        self.p_unc = p_unc
        self.p_fwer = p_fwer
        self.coordinate = Coordinate(
            str(peak_unique_id), label=coord_label, **kwargs)
        self.type = NIDM_PEAK
        self.prov_type = PROV['Entity']
        # self.cluster = cluster_id
        self.exc_set_id = exc_set_id

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_Peak: <http://purl.org/nidash/nidm#NIDM_0000062>
        prefix nidm_pValueUncorrected: <http://purl.org/nidash/nidm#NIDM_0000116>
        prefix nidm_equivalentZStatistic: <http://purl.org/nidash/nidm#NIDM_0000092>
        prefix nidm_pValueFWER: <http://purl.org/nidash/nidm#NIDM_0000115>
        prefix nidm_qValueFDR: <http://purl.org/nidash/nidm#NIDM_0000119>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_Peak: ;
                rdfs:label ?label .

            OPTIONAL {""" + oid_var + """ prov:value ?value .} .
            OPTIONAL {""" + oid_var + """ nidm_pValueUncorrected: ?p_unc .} .
            OPTIONAL {""" + oid_var + """ nidm_equivalentZStatistic: ?equiv_z .} .
            OPTIONAL {""" + oid_var + """ nidm_pValueFWER: ?p_fwer .} .
            OPTIONAL {""" + oid_var + """ nidm_qValueFDR: ?p_fdr .} .
        }
        """
        return query

    def __str__(self):
        return '%s \tz=%.2f \tp=%.2e (unc.) \t%s' % (
            self.label, self.equiv_z, self.p_unc, str(self.coordinate))

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        if self.p_unc is None:
            norm_cdf_z = (1.0 + erf(self.equiv_z / sqrt(2.0))) / 2.0
            self.p_unc = 1 - norm_cdf_z

        self.add_attributes([
            (PROV['type'], self.type),
            (PROV['label'], self.label),
            (NIDM_EQUIVALENT_ZSTATISTIC,
                Literal(self.equiv_z, datatype=XSD_FLOAT)),
            (NIDM_P_VALUE_UNCORRECTED,
                Literal(self.p_unc, datatype=XSD_FLOAT)),
            (PROV['location'], self.coordinate.id)])
