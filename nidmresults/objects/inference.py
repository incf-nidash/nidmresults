"""
Objects describing the Inference activity, its inputs and outputs as specified
in NIDM-Results.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""
from prov.model import Identifier
import os
from constants import *
import shutil
from generic import *
from scipy.stats import norm
import uuid


class Inference(NIDMObject):

    """
    Object representing an Inference step: including an Inference activity, its
    inputs and outputs.
    """

    def __init__(self, inference, height_thresh, extent_thresh, peak_criteria,
                 cluster_criteria, disp_mask, excursion_set, clusters,
                 search_space, software_id):
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

    def export(self):
        """
        Create prov entities and activities.
        """
        # Excursion set
        self.p.update(self.excursion_set.export())

        # Height threshold
        self.p.update(self.height_thresh.export())

        # Extent threshold
        self.p.update(self.extent_thresh.export())

        # Inference activity
        self.p.update(self.inference_act.export())

        if self.clusters:
            # Peak Definition
            self.p.update(self.peak_criteria.export())
            self.p.used(self.inference_act.id, self.peak_criteria.id)

            # Display Mask
            self.p.update(self.disp_mask.export())
            self.p.used(self.inference_act.id, self.disp_mask.id)

            # Search Space
            self.p.update(self.search_space.export())
            self.p.wasGeneratedBy(self.search_space.id, self.inference_act.id)

            # Cluster Definition
            self.p.update(self.cluster_criteria.export())
            self.p.used(self.inference_act.id, self.cluster_criteria.id)

            # Clusters and peaks
            for cluster in self.clusters:
                self.p.update(cluster.export())
                self.p.wasDerivedFrom(cluster.id, self.excursion_set.id)

        self.p.wasGeneratedBy(self.excursion_set.id, self.inference_act.id)

        self.p.wasAssociatedWith(self.inference_act.id, self.software_id)
        # self.p.wasGeneratedBy(NIIRI['search_space_id'],
            # self.inference_act.id)
        self.p.used(self.inference_act.id, self.height_thresh.id)
        self.p.used(self.inference_act.id, self.extent_thresh.id)
        # self.p.used(self.inference_act.id, NIIRI['z_statistic_map_id_'+
            # contrast_num])
        # self.p.used(self.inference_act.id, NIIRI['mask_id_1'])

        return self.p


class InferenceActivity(NIDMObject):

    """
    Object representing an Inference activity.
    """

    def __init__(self, contrast_num, contrast_name):
        super(InferenceActivity, self).__init__()
        self.id = NIIRI[str(uuid.uuid4())]
        self.contrast_name = contrast_name

    def export(self):
        """
        Create prov entities and activities.
        """

        label = "Inference: " + self.contrast_name
        # In FSL we have a single thresholding (extent, height) applied to all
        # contrasts
        # FIXME: Deal with two-tailed inference?
        self.p.activity(self.id,
                        other_attributes=((PROV['type'], NIDM_INFERENCE),
                                          (PROV['label'], label),
                                          (NIDM_HAS_ALTERNATIVE_HYPOTHESIS,
                                           NIDM_ONE_TAILED_TEST)))
        return self.p


class ExcursionSet(NIDMObject):

    """
    Object representing a ExcursionSet entity.
    """

    def __init__(self, filename, stat_num, visualisation, coord_space,
                 export_dir):
        super(ExcursionSet, self).__init__(export_dir)
        self.num = stat_num
        self.file = filename
        self.id = NIIRI[str(uuid.uuid4())]
        self.visu = Visualisation(visualisation, stat_num, export_dir)

        self.coord_space = coord_space

    def export(self):
        """
        Create prov entities and activities.
        """
        self.p.update(self.coord_space.export())

        self.p.update(self.visu.export())

        # Copy "Excursion set map" in export directory
        exc_set_orig_file = self.file
        exc_set_file = os.path.join(self.export_dir, 'ExcursionSet' +
                                    self.num + '.nii.gz')
        exc_set_orig_filename, exc_set_filename = self.copy_nifti(
            exc_set_orig_file, exc_set_file)

        # Create "Excursion set" entity
        self.p.entity(self.id, other_attributes=(
            (PROV['type'], NIDM_EXCURSION_SET_MAP),
            (DCT['format'], "image/nifti"),
            (PROV['location'], Identifier("file://./" + exc_set_filename)),
            (NFO['fileName'], exc_set_orig_filename),
            (NFO['fileName'], exc_set_filename),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], "Excursion Set Map"),
            (DC['description'], self.visu.id),
            (CRYPTO['sha512'], self.get_sha_sum(exc_set_file)),
        ))

        return self.p

# FIXME: When we decide how to compute URIs, this will need to be replaced by
# Image


class Visualisation(NIDMObject):

    """
    Object representing an Image entity.
    """

    def __init__(self, visu_filename, stat_num, export_dir):
        super(Visualisation, self).__init__(export_dir)
        self.file = visu_filename
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entities and activities.
        """
        # Copy visualisation of excursion set in export directory
        shutil.copy(self.file, self.export_dir)
        path, visu_filename = os.path.split(self.file)

        # Create "png visualisation of Excursion set" entity
        self.p.entity(self.id, other_attributes=(
            (PROV['type'], DCTYPE['Image']),
            (NFO['fileName'], visu_filename),
            (PROV['location'], Identifier("file://./" + visu_filename)),
            (DCT['format'], "image/png"),
        ))

        return self.p


class HeightThreshold(NIDMObject):

    """
    Object representing a HeightThreshold entity.
    """

    def __init__(self, stat_threshold=None, p_corr_threshold=None,
                 p_uncorr_threshold=None):
        super(HeightThreshold, self).__init__()
        if not stat_threshold and not p_corr_threshold and \
           not p_uncorr_threshold:
            raise Exception('No threshold defined')

        self.stat_threshold = stat_threshold
        self.p_corr_threshold = p_corr_threshold
        self.p_uncorr_threshold = p_uncorr_threshold
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entities and activities.
        """
        thresh_desc = ""
        if self.stat_threshold is not None:
            thresh_desc = "Z>" + str(self.stat_threshold)
            user_threshold_type = "Z-Statistic"
        elif self.p_uncorr_threshold is not None:
            thresh_desc = "p<" + \
                str(self.p_uncorr_threshold) + " (uncorrected)"
            user_threshold_type = "p-value uncorrected"
        elif self.p_corr_threshold is not None:
            thresh_desc = "p<" + str(self.p_corr_threshold) + " (FWE)"
            user_threshold_type = "p-value FWE"

        # FIXME: Do we want to calculate an uncorrected p equivalent to the Z
        # thresh?
        # FIXME: Do we want/Can we find a corrected p equivalent to the Z
        # thresh?
        heightThreshAllFields = {
            PROV['type']: NIDM_HEIGHT_THRESHOLD,
            PROV['label']: "Height Threshold: " + thresh_desc,
            NIDM_USER_SPECIFIED_THRESHOLD_TYPE: user_threshold_type,
            PROV['value']: self.stat_threshold,
            NIDM_P_VALUE_UNCORRECTED: self.p_uncorr_threshold,
            NIDM_P_VALUE_FWER: self.p_corr_threshold
        }
        self.p.entity(self.id, other_attributes=dict(
            (k, v) for k, v in heightThreshAllFields.iteritems()
            if v is not None))

        return self.p


class ExtentThreshold(NIDMObject):

    """
    Object representing an ExtentThreshold entity.
    """

    def __init__(self, extent=None, p_corr=None, p_uncorr=None):
        super(ExtentThreshold, self).__init__()
        self.extent = extent
        self.p_corr = p_corr
        self.p_uncorr = p_uncorr
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entities and activities.
        """
        thresh_desc = ""
        if self.extent is not None:
            thresh_desc = "k>" + str(self.extent)
            user_threshold_type = "Cluster-size in voxels"
        elif not self.p_uncorr is None:
            thresh_desc = "p<" + str(self.p_uncorr) + " (uncorrected)"
            user_threshold_type = "p-value uncorrected"
        elif not self.p_corr is None:
            thresh_desc = "p<" + str(self.p_corr) + " (FWE)"
            user_threshold_type = "p-value FWE"
        else:
            thresh_desc = "k>=0"
            self.extent = 0
            self.p_uncorr = 1.0
            self.p_corr = 1.0
            user_threshold_type = None

        extent_thresh_all_fields = {
            PROV['type']: NIDM_EXTENT_THRESHOLD,
            PROV['label']: "Extent Threshold: " + thresh_desc,
            NIDM_CLUSTER_SIZE_IN_VOXELS: self.extent,
            NIDM_USER_SPECIFIED_THRESHOLD_TYPE: user_threshold_type,
            NIDM_P_VALUE_UNCORRECTED: self.p_uncorr,
            NIDM_P_VALUE_FWER: self.p_corr
        }
        self.p.entity(self.id, other_attributes=dict(
            (k, v) for k, v in extent_thresh_all_fields.iteritems()
            if v is not None))

        return self.p


class Cluster(NIDMObject):

    """
    Object representing a Cluster entity.
    """

    def __init__(self, cluster_num, size, pFWER, peaks,
                 x=None, y=None, z=None, x_std=None, y_std=None, z_std=None):
        super(Cluster, self).__init__()
        self.num = cluster_num
        self.id = NIIRI[str(uuid.uuid4())]
        self.cog = CenterOfGravity(
            cluster_num, x=x, y=y, z=z, x_std=x_std, y_std=y_std, z_std=z_std)
        self.peaks = peaks
        self.size = size
        self.pFWER = pFWER

    def export(self):
        """
        Create prov entities and activities.
        """
        for peak in self.peaks:
            self.p.update(peak.export())
            self.p.wasDerivedFrom(peak.id, self.id)

        self.p.update(self.cog.export())
        self.p.wasDerivedFrom(self.cog.id, self.id)

        # FIXME deal with multiple contrasts
        self.p.entity(self.id, other_attributes=(
            (PROV['type'], NIDM_SIGNIFICANT_CLUSTER),
            (PROV['label'], "Significant Cluster %04d" % self.num),
            (NIDM_CLUSTER_LABEL_ID, self.num),
            (NIDM_CLUSTER_SIZE_IN_VOXELS, self.size),
            (NIDM_P_VALUE_FWER, self.pFWER)))
        return self.p


class DisplayMaskMap(NIDMObject):

    """
    Object representing a DisplayMaskMap entity.
    """

    def __init__(self, contrast_num, filename, coord_space, export_dir):
        super(DisplayMaskMap, self).__init__(export_dir)
        self.id = NIIRI[str(uuid.uuid4())]
        self.filename = filename
        self.coord_space = coord_space

    def export(self):
        """
        Create prov entities and activities.
        """
        # Create coordinate space entity
        self.p.update(self.coord_space.export())

        # Create "Display Mask Map" entity
        disp_mask_file = os.path.join(self.export_dir, 'DisplayMask.nii.gz')
        disp_mask_orig_filename, disp_mask_filename = self.copy_nifti(
            self.filename, disp_mask_file)

        self.p.entity(self.id,
                      other_attributes=(
                          (PROV['type'], NIDM_DISPLAY_MASK_MAP),
                          (PROV['label'], "Display Mask Map"),
                          (DCT['format'], "image/nifti"),
                          (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
                          (NFO['fileName'], disp_mask_orig_filename),
                          (NFO['fileName'], disp_mask_filename),
                          (PROV['location'],
                              Identifier("file://./" + disp_mask_filename)),
                          (CRYPTO['sha512'], self.get_sha_sum(disp_mask_file)))
                      )
        return self.p


class PeakCriteria(NIDMObject):

    """
    Object representing a PeakCriteria entity.
    """

    def __init__(self, contrast_num, num_peak, peak_dist):
        super(PeakCriteria, self).__init__()
        self.id = NIIRI[str(uuid.uuid4())]
        self.num_peak = num_peak
        self.peak_dist = peak_dist

    def export(self):
        """
        Create prov entities and activities.
        """
        num_peak = ()
        if self.num_peak:
            num_peak = (NIDM_MAXNUMBEROFPEAKSPERCLUSTER, self.num_peak)

        # Create "Peak definition criteria" entity
        self.p.entity(self.id,
                      other_attributes=(
                          (PROV['type'], NIDM_PEAK_DEFINITION_CRITERIA),
                          (PROV['label'], "Peak Definition Criteria"),
                          (NIDM_MIN_DISTANCE_BETWEEN_PEAKS, self.peak_dist)) +
                      num_peak)

        return self.p


class ClusterCriteria(NIDMObject):

    """
    Object representing a ClusterCriteria entity.
    """

    def __init__(self, contrast_num, connectivity):
        super(ClusterCriteria, self).__init__()
        self.id = NIIRI[str(uuid.uuid4())]
        self.connectivity = connectivity

    def export(self):
        """
        Create prov entities and activities.
        """
        # Create "Cluster definition criteria" entity
        if self.connectivity == 6:
            voxel_conn = NIDM_VOXEL6CONNECTED
        elif self.connectivity == 18:
            voxel_conn = NIDM_VOXEL18CONNECTED
        elif self.connectivity == 26:
            voxel_conn = NIDM_VOXEL26CONNECTED

        label = "Cluster Connectivity Criterion: " + str(self.connectivity)

        # FIXME if connectivity is missing
        if self.connectivity is not None:
            self.p.entity(self.id,
                          other_attributes=((PROV['type'],
                                             NIDM_CLUSTER_DEFINITION_CRITERIA),
                                            (PROV['label'], label),
                                            (NIDM_HAS_CONNECTIVITY_CRITERION,
                                                voxel_conn)))
        else:
            self.p.entity(
                self.id,
                other_attributes=(
                    (PROV['type'], NIDM_CLUSTER_DEFINITION_CRITERIA),
                    (PROV['label'], label)))

        return self.p


class CenterOfGravity(NIDMObject):

    """
    Object representing a CenterOfGravity entity.
    """

    def __init__(self, cluster_num, x=None, y=None, z=None, x_std=None,
                 y_std=None, z_std=None):
        super(CenterOfGravity, self).__init__()
        self.cluster_num = cluster_num
        self.id = NIIRI[str(uuid.uuid4())]
        self.coordinate = Coordinate("%04d" % cluster_num, x=x, y=y, z=z,
                                     x_std=x_std, y_std=y_std, z_std=z_std)

    def export(self):
        """
        Create prov entities and activities.
        """
        self.p.update(self.coordinate.export())

        label = "Center of gravity " + str(self.cluster_num)

        self.p.entity(self.id, other_attributes=(
                     (PROV['type'], NIDM_CLUSTER_CENTER_OF_GRAVITY),
                     (PROV['label'], label),
                     (PROV['location'], self.coordinate.id)))

        return self.p


class SearchSpace(NIDMObject):

    """
    Object representing a SearchSpace entity.
    """

    def __init__(self, search_space_file, vol_in_voxels, vol_in_units,
                 vol_in_resels, resel_size_in_voxels, dlh,
                 random_field_stationarity, noise_fwhm_in_voxels,
                 noise_fwhm_in_units, coord_space, export_dir):
        super(SearchSpace, self).__init__(export_dir)
        self.file = search_space_file
        self.coord_space = coord_space
        self.resel_size_in_voxels = resel_size_in_voxels
        self.dlh = dlh
        self.search_volume_in_voxels = vol_in_voxels
        self.search_volume_in_units = vol_in_units
        self.search_volume_in_resels = vol_in_resels
        self.rf_stationarity = random_field_stationarity
        self.noise_fwhm_in_voxels = noise_fwhm_in_voxels
        self.noise_fwhm_in_units = noise_fwhm_in_units
        self.id = NIIRI[str(uuid.uuid4())]

    # Generate prov for search space entity generated by the inference activity
    def export(self):
        """
        Create prov entities and activities.
        """
        self.p.update(self.coord_space.export())

        # Copy "Mask map" in export directory
        search_space_orig_file = self.file
        search_space_file = os.path.join(self.export_dir,
                                         'SearchSpaceMask.nii.gz')
        search_space_orig_filename, search_space_filename = self.copy_nifti(
            search_space_orig_file, search_space_file)

        # Crate "Mask map" entity
        self.p.entity(self.id, other_attributes=(
            (PROV['label'], "Search Space Mask Map"),
            (DCT['format'], "image/nifti"),
            (PROV['type'], NIDM_SEARCH_SPACE_MASK_MAP),
            (PROV['location'],
             Identifier("file://./" + search_space_filename)),
            (NFO['fileName'], search_space_orig_filename),
            (NFO['fileName'], search_space_filename),
            (NIDM_RANDOM_FIELD_STATIONARITY, self.rf_stationarity),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (NIDM_SEARCH_VOLUME_IN_VOXELS, self.search_volume_in_voxels),
            (NIDM_SEARCH_VOLUME_IN_UNITS, self.search_volume_in_units),
            (NIDM_SEARCH_VOLUME_IN_RESELS, self.search_volume_in_resels),
            (SPM_NOISE_FWHM_IN_VOXELS, self.noise_fwhm_in_voxels),
            (SPM_NOISE_FWHM_IN_UNITS, self.noise_fwhm_in_units),
            (CRYPTO['sha512'], self.get_sha_sum(search_space_file)),
            (NIDM_RESEL_SIZE_IN_VOXELS, self.resel_size_in_voxels),
            (NIDM_NOISE_ROUGHNESS_IN_VOXELS, self.dlh)))

        return self.p


class Coordinate(NIDMObject):

    """
    Object representing a Coordinate entity.
    """

    def __init__(self, label_id, coord_vector=None, coord_vector_std=None,
                 x=None, y=None, z=None, x_std=None, y_std=None, z_std=None):
        super(Coordinate, self).__init__()
        # FIXME: coordiinate_id should not be determined externally
        self.id = NIIRI[str(uuid.uuid4())]
        self.label_id = label_id
        if x is not None and y is not None and z is not None:
            self.coord_vector = \
                "[ " + str(x) + ", " + str(y) + ", " + str(z) + " ]"
        else:
            self.coord_vector = coord_vector
        if x_std is not None and y_std is not None and z_std is not None:
            self.coord_vector_std = \
                "[ " + str(x_std) + ", " + str(y_std) + ", " + \
                str(z_std) + " ]"
        else:
            self.coord_vector_std = coord_vector_std

    def export(self):
        """
        Create prov entities and activities.
        """
        # We can not have this in the dictionnary because we want to keep the
        # duplicate prov:type attribute
        typeAndLabelAttributes = [  # (PROV['type'],PROV['Location']),
            (PROV['type'], NIDM_COORDINATE),
            (PROV['label'], "Coordinate " + self.label_id)]

        coordinateAttributes = {
            NIDM_COORDINATE_VECTOR_IN_VOXELS: self.coord_vector,
            NIDM_COORDINATE_VECTOR: self.coord_vector_std,
        }

        self.p.entity(
            self.id,
            other_attributes=typeAndLabelAttributes +
            list(dict((k, v) for k, v in coordinateAttributes.iteritems()
                      if not v is None).items()))

        return self.p


class Peak(NIDMObject):

    """
    Object representing a Peak entity.
    """

    def __init__(self, cluster_index, peak_index, equiv_z, stat_num,
                 *args, **kwargs):
        super(Peak, self).__init__()
        # FIXME: Currently assumes less than 10 clusters per contrast
        # cluster_num = cluster_index
        # FIXME: Currently assumes less than 100 peaks
        peak_unique_id = '000' + str(cluster_index) + '_' + str(peak_index)
        self.id = NIIRI[str(uuid.uuid4())]
        self.num = peak_unique_id
        self.equiv_z = equiv_z
        self.coordinate = Coordinate(str(peak_unique_id), **kwargs)

    def export(self):
        """
        Create prov entities and activities.
        """
        self.p.update(self.coordinate.export())

        other_attributes = [
            (PROV['type'], NIDM_PEAK),
            (PROV['label'], "Peak " + str(self.num)),
            (NIDM_EQUIVALENT_ZSTATISTIC, self.equiv_z),
            (NIDM_P_VALUE_UNCORRECTED, 1 - norm.cdf(self.equiv_z)),
            (PROV['location'], self.coordinate.id)]

        self.p.entity(self.id, other_attributes=other_attributes)

        return self.p
