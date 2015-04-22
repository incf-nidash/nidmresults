"""
Objects describing the Model Parameters Estimation activity, its inputs and
outputs as specified in NIDM-Results.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""
from prov.model import Identifier
import uuid
import numpy as np
import os
from constants import *
import nibabel as nib
from generic import *


class ModelFitting(NIDMObject):

    """
    Object representing a Model fitting step: including a
    ModelParametersEstimation activity, its inputs and outputs.
    """

    def __init__(self, activity, design_matrix, data, error_model,
                 param_estimates, rms_map, mask_map, grand_mean_map):
        super(ModelFitting, self).__init__()
        self.activity = activity
        self.design_matrix = design_matrix
        self.data = data
        self.error_model = error_model
        self.param_estimates = param_estimates
        self.rms_map = rms_map
        self.mask_map = mask_map
        self.grand_mean_map = grand_mean_map

    def export(self):
        """
        Create prov entities and activities.
        """
        # Model Parameters Estimation activity
        self.p.update(self.activity.export())

        # Design Matrix
        self.p.update(self.design_matrix.export())
        self.p.used(self.activity.id, self.design_matrix.id)

        # Data
        self.p.update(self.data.export())
        self.p.used(self.activity.id, self.data.id)

        # Error Model
        self.p.update(self.error_model.export())
        self.p.used(self.activity.id, self.error_model.id)

        # Parameter Estimate Maps
        for param_estimate in self.param_estimates:
            self.p.update(param_estimate.export())
            self.p.wasGeneratedBy(param_estimate.id, self.activity.id)

        # Residual Mean Squares Map
        self.p.update(self.rms_map.export())
        self.p.wasGeneratedBy(self.rms_map.id, self.activity.id)

        # Mask
        self.p.update(self.mask_map.export())
        self.p.wasGeneratedBy(self.mask_map.id, self.activity.id)

        # Grand Mean map
        self.p.update(self.grand_mean_map.export())
        self.p.wasGeneratedBy(self.grand_mean_map.id, self.activity.id)

        return self.p


class DesignMatrix(NIDMObject):

    """
    Object representing a DesignMatrix entity.
    """

    def __init__(self, matrix, image_file, export_dir):
        super(DesignMatrix, self).__init__(export_dir=export_dir)
        self.matrix = matrix
        self.id = NIIRI[str(uuid.uuid4())]
        self.image = Image(export_dir, image_file)

    def export(self):
        """
        Create prov entities and activities.
        """
        # Export visualisation of the design matrix
        self.p.update(self.image.export())

        # Create cvs file containing design matrix
        design_matrix_csv = 'DesignMatrix.csv'
        np.savetxt(os.path.join(self.export_dir, design_matrix_csv),
                   np.asarray(self.matrix), delimiter=",")

        # Create "design matrix" entity
        self.p.entity(
            self.id,
            other_attributes=((PROV['type'], NIDM['DesignMatrix']),
                              (PROV['label'], "Design Matrix"),
                              (DCT['format'], "text/csv"),
                              (NIDM['filename'], "DesignMatrix.csv"),
                              (NIDM['visualisation'], self.image.id),
                              (PROV['location'],
                               Identifier("file://./" + design_matrix_csv))))
        return self.p


class Data(NIDMObject):

    """
    Object representing a Data entity.
    """

    def __init__(self, grand_mean_scaling, target):
        super(Data, self).__init__()
        self.grand_mean_sc = grand_mean_scaling
        self.target_intensity = target
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entities and activities.
        """
    # Create "Data" entity
        # FIXME: grand mean scaling?
        # FIXME: medianIntensity
        self.p.entity(self.id,
                      other_attributes=((PROV['type'], NIDM['Data']),
                                        (PROV['type'], PROV['Collection']),
                                        (PROV['label'], "Data"),
                                        (NIDM['grandMeanScaling'],
                                         self.grand_mean_sc),
                                        (NIDM['targetIntensity'],
                                         self.target_intensity)))
        return self.p


class ErrorModel(NIDMObject):

    """
    Object representing an ErrorModel entity.
    """

    def __init__(self, error_distribution, variance_homo, variance_spatial,
                 dependance, dependance_spatial):
        super(ErrorModel, self).__init__()
        self.error_distribution = error_distribution
        self.variance_homo = variance_homo
        self.variance_spatial = variance_spatial
        self.dependance = dependance
        self.dependance_spatial = dependance_spatial
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entities and activities.
        """
        # Create "Error Model" entity
        self.p.entity(self.id,
                      other_attributes=((PROV['type'], NIDM['ErrorModel']),
                                        (NIDM['hasErrorDistribution'],
                                         self.error_distribution),
                                        (NIDM['errorVarianceHomogeneous'],
                                         self.variance_homo),
                                        (NIDM['varianceSpatialModel'],
                                         self.variance_spatial),
                                        (NIDM['hasErrorDependence'],
                                         self.dependance),
                                        (NIDM['dependenceSpatialModel'],
                                         self.dependance_spatial)))

        return self.p


class ModelParametersEstimation(NIDMObject):

    """
    Object representing an ModelParametersEstimation activity.
    """

    def __init__(self, estimation_method, software_id):
        super(ModelParametersEstimation, self).__init__()
        self.estimation_method = estimation_method
        self.software_id = software_id
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entities and activities.
        """
        # Create "Model Parameter estimation" activity
        self.p.activity(self.id, other_attributes=(
            (PROV['type'], NIDM['ModelParametersEstimation']),
            (NIDM['withEstimationMethod'], self.estimation_method),
            (PROV['label'], "Model Parameters Estimation")))

        return self.p


class ParameterEstimateMap(NIDMObject):

    """
    Object representing an ParameterEstimateMap entity.
    """

    def __init__(self, filename, pe_num, coord_space):
        super(ParameterEstimateMap, self).__init__()
        self.file = filename
        # Column index in the corresponding design matrix
        self.num = pe_num
        self.coord_space = coord_space
        self.id = NIIRI[str(uuid.uuid4())]

    # Generate prov for contrast map
    def export(self):
        """
        Create prov entities and activities.
        """
        self.p.update(self.coord_space.export())

        # Copy parameter estimate map in export directory
        # shutil.copy(pe_file, self.export_dir)
        path, pe_filename = os.path.split(self.file)
        # pe_file = os.path.join(self.export_dir,pe_filename)

        # Parameter estimate entity
        self.p.entity(
            self.id,
            other_attributes=((PROV['type'], NIDM['ParameterEstimateMap']),
                              # (DCT['format'], "image/nifti"),
                              # (PROV['location'],
                              # Identifier("file://./"+pe_filename)),
                              (NIDM['filename'], pe_filename),
                              (NIDM['inCoordinateSpace'],
                               self.coord_space.id),
                              # (CRYPTO['sha512'], self.get_sha_sum(pe_file)),
                              (PROV['label'],
                               "Parameter estimate " + str(self.num))))

        return self.p


class ResidualMeanSquares(NIDMObject):

    """
    Object representing an ResidualMeanSquares entity.
    """

    def __init__(self, export_dir, residual_file, coord_space):
        super(ResidualMeanSquares, self).__init__(export_dir)
        self.file = residual_file
        self.coord_space = coord_space
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entities and activities.
        """
        # Create coordinate space export
        self.p.update(self.coord_space.export())

        # Copy residuals map in export directory
        residuals_file = os.path.join(
            self.export_dir, 'ResidualMeanSquares.nii.gz')
        residuals_original_filename, residuals_filename = self.copy_nifti(
            self.file, residuals_file)

        # Create "residuals map" entity
        self.p.entity(
            self.id,
            other_attributes=(
                (PROV['type'], NIDM['ResidualMeanSquaresMap'],),
                (DCT['format'], "image/nifti"),
                (PROV['location'],
                 Identifier("file://./" + residuals_filename)),
                (PROV['label'],
                 "Residual Mean Squares Map"),
                (NIDM['filename'],
                 residuals_original_filename),
                (NIDM['filename'], residuals_filename),
                (CRYPTO['sha512'],
                 self.get_sha_sum(residuals_file)),
                (NIDM['inCoordinateSpace'], self.coord_space.id)))

        return self.p


class MaskMap(NIDMObject):

    """
    Object representing an MaskMap entity.
    """

    def __init__(self, export_dir, mask_file, coord_space):
        super(MaskMap, self).__init__(export_dir)
        self.file = mask_file
        self.coord_space = coord_space
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entities and activities.
        """
        # Create coordinate space export
        self.p.update(self.coord_space.export())

        # Create "Mask map" entity
        original_mask_file = self.file
        mask_file = os.path.join(self.export_dir, 'Mask.nii.gz')

        original_mask_filename, mask_filename = self.copy_nifti(
            original_mask_file, mask_file)
        self.p.entity(
            self.id,
            other_attributes=(
                (PROV['type'], NIDM['MaskMap'],),
                (DCT['format'], "image/nifti"),
                (PROV['location'],
                 Identifier("file://./" + mask_filename)),
                (PROV['label'], "Mask"),
                (NIDM['filename'],
                 original_mask_filename),
                (NIDM['filename'], mask_filename),
                (CRYPTO['sha512'],
                 self.get_sha_sum(mask_file)),
                (NIDM['inCoordinateSpace'], self.coord_space.id)))

        return self.p


class GrandMeanMap(NIDMObject):

    """
    Object representing an GrandMeanMap entity.
    """

    def __init__(self, filename, mask_file, coord_space, export_dir):
        super(GrandMeanMap, self).__init__(export_dir)
        self.id = NIIRI[str(uuid.uuid4())]
        self.file = filename
        self.mask_file = mask_file  # needed to compute masked median
        self.coord_space = coord_space

    def export(self):
        """
        Create prov entities and activities.
        """
        # Coordinate space entity
        self.p.update(self.coord_space.export())

        # Grand Mean Map entity
        grand_mean_file = os.path.join(self.export_dir, 'GrandMean.nii.gz')
        grand_mean_original_filename, grand_mean_filename = self.copy_nifti(
            self.file, grand_mean_file)
        grand_mean_img = nib.load(grand_mean_file)
        grand_mean_data = grand_mean_img.get_data()
        grand_mean_data = np.ndarray.flatten(grand_mean_data)

        mask_img = nib.load(self.mask_file)
        mask_data = mask_img.get_data()
        mask_data = np.ndarray.flatten(mask_data)

        grand_mean_data_in_mask = grand_mean_data[mask_data > 0]
        masked_median = np.median(
            np.array(grand_mean_data_in_mask, dtype=float))

        self.p.entity(
            self.id,
            other_attributes=((PROV['type'], NIDM['GrandMeanMap']),
                              (DCT['format'], "image/nifti"),
                              (PROV['label'], "Grand Mean Map"),
                              (NIDM['maskedMedian'], masked_median),
                              (NIDM['filename'],
                               grand_mean_filename),
                              (NIDM['filename'],
                               grand_mean_original_filename),
                              (NIDM['inCoordinateSpace'],
                               self.coord_space.id),
                              (CRYPTO['sha512'],
                               self.get_sha_sum(grand_mean_file)),
                              (PROV['location'],
                               Identifier("file://./" + grand_mean_filename))))

        return self.p
