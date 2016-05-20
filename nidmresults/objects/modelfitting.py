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
import json
import warnings


class ModelFitting(NIDMObject):

    """
    Object representing a Model fitting step: including a
    ModelParametersEstimation activity, its inputs and outputs.
    """

    def __init__(self, activity, design_matrix, data, error_model,
                 param_estimates, rms_map, mask_map, grand_mean_map,
                 machine, subjects):
        super(ModelFitting, self).__init__()
        self.activity = activity
        self.design_matrix = design_matrix
        self.data = data
        self.error_model = error_model
        self.param_estimates = param_estimates
        self.rms_map = rms_map
        self.mask_map = mask_map
        self.grand_mean_map = grand_mean_map
        self.machine = machine
        self.subjects = subjects

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        # Design Matrix
        self.activity.used(self.design_matrix)
        self.add_object(self.design_matrix, nidm_version)

        if nidm_version['major'] > 1 or \
                (nidm_version['major'] == 1 and nidm_version['minor'] >= 3):
            # Machine
            self.data.wasAttributedTo(self.machine)
            self.add_object(self.machine, nidm_version)

            # Imaged subject or group(s)
            for sub in self.subjects:
                self.add_object(sub, nidm_version)
                self.data.wasAttributedTo(sub)

        # Data
        self.activity.used(self.data)
        self.add_object(self.data, nidm_version)

        # Error Model
        self.activity.used(self.error_model)
        self.add_object(self.error_model, nidm_version)

        # Parameter Estimate Maps
        for param_estimate in self.param_estimates:
            param_estimate.wasGeneratedBy(self.activity)
            self.add_object(param_estimate, nidm_version)

        # Residual Mean Squares Map
        self.rms_map.wasGeneratedBy(self.activity)
        self.add_object(self.rms_map, nidm_version)

        # Mask
        self.mask_map.wasGeneratedBy(self.activity)
        self.add_object(self.mask_map, nidm_version)

        # Grand Mean map
        self.grand_mean_map.wasGeneratedBy(self.activity)
        self.add_object(self.grand_mean_map, nidm_version)

        # Model Parameters Estimation activity
        self.add_object(self.activity, nidm_version)

        return self.p


class ImagingInstrument(NIDMObject):
    """
    Object representing a ImagingInstrument entity.
    """

    def __init__(self, machine_type):
        super(ImagingInstrument, self).__init__()
        machine_type = machine_type.lower()
        self.id = NIIRI[str(uuid.uuid4())]
        machine_term = dict(
            mri=NIF_MRI, eeg=NIF_EEG, meg=NIF_MEG, pet=NIF_PET,
            spect=NIF_SPECT)
        machine_label = dict(
            mri='MRI Scanner', eeg='EEG Machine', meg='MEG Machine',
            pet='PET Scanner', spect='SPECT Machine')
        self.type = machine_term[machine_type]
        self.prov_type = PROV['Agent']
        self.label = machine_label[machine_type]

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['label'], self.label)))

        return self.p


class Group(NIDMObject):
    """
    Object representing a ImagingInstrument entity.
    """

    def __init__(self, num_subjects, group_name):
        super(Group, self).__init__()
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = STATO_GROUP
        self.prov_type = PROV['Agent']
        self.group_name = group_name
        self.num_subjects = num_subjects

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_GROUP_NAME, self.group_name),
            (NIDM_NUMBER_OF_SUBJECTS, self.num_subjects),
            (PROV['label'], "Study group population")))

        return self.p


class Person(NIDMObject):
    """
    Object representing a ImagingInstrument entity.
    """

    def __init__(self):
        super(Person, self).__init__()
        self.id = NIIRI[str(uuid.uuid4())]
        self.prov_type = PROV['Agent']
        self.type = PROV['Person']

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.prov_type),
            (PROV['label'], "Person")))

        return self.p


class DesignMatrix(NIDMObject):

    """
    Object representing a DesignMatrix entity.
    """

    def __init__(self, matrix, image_file, export_dir, regressors=None,
                 design_type=None, hrf_model=None, drift_model=None):
        super(DesignMatrix, self).__init__(export_dir=export_dir)
        self.type = NIDM_DESIGN_MATRIX
        self.prov_type = PROV['Entity']
        self.matrix = matrix
        self.id = NIIRI[str(uuid.uuid4())]
        self.image = Image(export_dir, image_file)
        self.regressors = regressors
        self.design_type = design_type
        self.hrf_model = hrf_model
        self.drift_model = drift_model

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        # *** Export visualisation of the design matrix
        self.add_object(self.image, nidm_version)

        # Create cvs file containing design matrix
        design_matrix_csv = 'DesignMatrix.csv'
        np.savetxt(os.path.join(self.export_dir, design_matrix_csv),
                   np.asarray(self.matrix), delimiter=",")

        if nidm_version['num'] in ["1.0.0", "1.1.0"]:
            csv_location = Identifier("file://./" + design_matrix_csv)
        else:
            csv_location = Identifier(design_matrix_csv)

        attributes = [(PROV['type'], self.type),
                      (PROV['label'], "Design Matrix"),
                      (NIDM_REGRESSOR_NAMES, json.dumps(self.regressors)),
                      (DCT['format'], "text/csv"),
                      (NFO['fileName'], "DesignMatrix.csv"),
                      (DC['description'], self.image.id),
                      (PROV['location'], csv_location)]

        if self.hrf_model is not None:
            if nidm_version['num'] in ("1.0.0", "1.1.0"):
                if self.design_type is not None:
                    attributes.append(
                        (NIDM_HAS_FMRI_DESIGN, self.design_type))
                else:
                    warnings.warn("Design type is missing")

            # hrf model
            attributes.append((NIDM_HAS_HRF_BASIS, self.hrf_model))
            # drift model
            self.add_object(self.drift_model, nidm_version)
            attributes.append((NIDM_HAS_DRIFT_MODEL, self.drift_model.id))

        # Create "design matrix" entity
        self.add_attributes(attributes)

        return self.p


class DriftModel(NIDMObject):

    """
    Object representing a DriftModel entity.
    """

    def __init__(self, drift_type, parameter):
        super(DriftModel, self).__init__()
        self.drift_type = drift_type
        self.id = NIIRI[str(uuid.uuid4())]
        self.parameter = parameter
        self.type = drift_type
        self.prov_type = PROV['Entity']

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        attributes = [(PROV['type'], self.drift_type)]

        if self.drift_type == FSL_GAUSSIAN_RUNNING_LINE_DRIFT_MODEL:
            attributes.append(
                (PROV['label'], "FSL's Gaussian Running Line Drift Model"))
            attributes.append((FSL_DRIFT_CUTOFF_PERIOD, self.parameter))

        # Create "drift model" entity
        self.add_attributes(attributes)

        return self.p


class Data(NIDMObject):

    """
    Object representing a Data entity.
    """

    def __init__(self, grand_mean_scaling, target, mri_protocol=None):
        super(Data, self).__init__()
        self.grand_mean_sc = grand_mean_scaling
        self.target_intensity = target
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = NIDM_DATA
        self.prov_type = PROV['Entity']
        self.mri_protocol = mri_protocol
        if self.mri_protocol == "fmri":
            self.mri_protocol = NLX_FMRI_PROTOCOL

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        if nidm_version['major'] < 1 or \
                (nidm_version['major'] == 1 and nidm_version['minor'] < 3):
            self.type = NIDM_DATA_SCALING
    # Create "Data" entity
        # FIXME: grand mean scaling?
        # FIXME: medianIntensity
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['type'], PROV['Collection']),
            (PROV['label'], "Data"),
            (NIDM_GRAND_MEAN_SCALING, self.grand_mean_sc),
            (NIDM_TARGET_INTENSITY, self.target_intensity)))

        if nidm_version['major'] > 1 or \
                (nidm_version['major'] == 1 and nidm_version['minor'] > 2):
            if self.mri_protocol is not None:
                self.add_attributes(
                    [(NIDM_HAS_MRI_PROTOCOL, self.mri_protocol)])

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
        self.type = NIDM_ERROR_MODEL
        self.prov_type = PROV['Entity']

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        atts = (
            (PROV['type'], NIDM_ERROR_MODEL),
            (NIDM_HAS_ERROR_DISTRIBUTION, self.error_distribution),
            (NIDM_ERROR_VARIANCE_HOMOGENEOUS, self.variance_homo),
            (NIDM_VARIANCE_SPATIAL_MODEL, self.variance_spatial),
            (NIDM_HAS_ERROR_DEPENDENCE, self.dependance))

        # If the error covariance is independent then there is no associated
        # spatial model
        if self.dependance_spatial is not None:
            atts = atts + (
                ((NIDM_DEPENDENCE_SPATIAL_MODEL, self.dependance_spatial),))

        # Create "Error Model" entity
        self.add_attributes(atts)

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
        self.type = NIDM_MODEL_PARAMETERS_ESTIMATION
        self.prov_type = PROV['Activity']

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        # Create "Model Parameter estimation" activity
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_WITH_ESTIMATION_METHOD, self.estimation_method),
            (PROV['label'], "Model Parameters Estimation")))

        return self.p


class ParameterEstimateMap(NIDMObject):

    """
    Object representing an ParameterEstimateMap entity.
    """

    def __init__(self, pe_file, pe_num, coord_space):
        super(ParameterEstimateMap, self).__init__()
        # Column index in the corresponding design matrix
        self.num = pe_num
        self.coord_space = coord_space
        self.id = NIIRI[str(uuid.uuid4())]
        self.file = NIDMFile(self.id, pe_file)
        self.type = NIDM_PARAMETER_ESTIMATE_MAP
        self.prov_type = PROV['Entity']

    # Generate prov for contrast map
    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_object(self.coord_space, nidm_version)
        self.add_object(self.file, nidm_version)

        # Parameter estimate entity
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], "Parameter estimate " + str(self.num))))

        return self.p


class ResidualMeanSquares(NIDMObject):

    """
    Object representing an ResidualMeanSquares entity.
    """

    def __init__(self, export_dir, residual_file, coord_space,
                 temporary=False):
        super(ResidualMeanSquares, self).__init__(export_dir)
        self.coord_space = coord_space
        self.id = NIIRI[str(uuid.uuid4())]
        filename = 'ResidualMeanSquares.nii.gz'
        self.file = NIDMFile(self.id, residual_file, filename, export_dir,
                             temporary=temporary)
        self.type = NIDM_RESIDUAL_MEAN_SQUARES_MAP
        self.prov_type = PROV['Entity']

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        # Create coordinate space export
        self.add_object(self.coord_space, nidm_version)

        # Create "residuals map" entity
        self.add_object(self.file, nidm_version)

        self.add_attributes((
            (PROV['type'], self.type,),
            (PROV['label'], "Residual Mean Squares Map"),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id)))

        return self.p


class MaskMap(NIDMObject):

    """
    Object representing an MaskMap entity.
    """

    def __init__(self, export_dir, mask_file, coord_space, user_defined):
        super(MaskMap, self).__init__(export_dir)
        self.coord_space = coord_space
        self.id = NIIRI[str(uuid.uuid4())]
        filename = 'Mask.nii.gz'
        self.file = NIDMFile(self.id, mask_file, filename, export_dir)
        self.user_defined = user_defined
        self.type = NIDM_MASK_MAP
        self.prov_type = PROV['Entity']

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        # Create coordinate space export
        self.add_object(self.coord_space, nidm_version)

        # Create "Mask map" entity
        self.add_object(self.file, nidm_version)

        self.add_attributes((
            (PROV['type'], self.type,),
            (PROV['label'], "Mask"),
            (NIDM_IS_USER_DEFINED, self.user_defined),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id))
        )

        return self.p


class GrandMeanMap(NIDMObject):

    """
    Object representing an GrandMeanMap entity.
    """

    def __init__(self, org_file, mask_file, coord_space, export_dir):
        super(GrandMeanMap, self).__init__(export_dir)
        self.id = NIIRI[str(uuid.uuid4())]
        filename = 'GrandMean.nii.gz'
        self.file = NIDMFile(self.id, org_file, filename, export_dir)
        self.mask_file = mask_file  # needed to compute masked median
        self.coord_space = coord_space
        self.type = NIDM_GRAND_MEAN_MAP
        self.prov_type = PROV['Entity']

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        # Coordinate space entity
        self.add_object(self.coord_space, nidm_version)

        # Grand Mean Map entity
        self.add_object(self.file, nidm_version)

        grand_mean_file = self.file.path
        grand_mean_img = nib.load(grand_mean_file)
        grand_mean_data = grand_mean_img.get_data()
        grand_mean_data = np.ndarray.flatten(grand_mean_data)

        mask_img = nib.load(self.mask_file)
        mask_data = mask_img.get_data()
        mask_data = np.ndarray.flatten(mask_data)

        grand_mean_data_in_mask = grand_mean_data[mask_data > 0]
        masked_median = np.median(
            np.array(grand_mean_data_in_mask, dtype=float))

        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['label'], "Grand Mean Map"),
            (NIDM_MASKED_MEDIAN, masked_median),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id))
        )

        return self.p
