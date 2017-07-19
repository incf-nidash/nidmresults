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
from nidmresults.objects.constants import *
import nibabel as nib
from nidmresults.objects.generic import *
import json
import warnings
from numpy import genfromtxt
from prov.identifier import QualifiedName


class ModelFitting(object):

    """
    Object representing a Model fitting step: including a
    ModelParametersEstimation activity, its inputs and outputs.
    """

    def __init__(self, activity, design_matrix, data, error_model,
                 param_estimates, rms_map, mask_map, grand_mean_map,
                 machine, subjects):
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


class ImagingInstrument(NIDMObject):
    """
    Object representing a ImagingInstrument entity.
    """

    def __init__(self, machine_type, label=None, oid=None):
        super(ImagingInstrument, self).__init__(oid=oid)
        machine_type = machine_type.lower()
        self.id = NIIRI[str(uuid.uuid4())]
        machine_term = dict(
            mri=NIF_MRI, eeg=NIF_EEG, meg=NIF_MEG, pet=NIF_PET,
            spect=NIF_SPECT)
        machine_label = dict(
            mri='MRI Scanner', eeg='EEG Machine', meg='MEG Machine',
            pet='PET Scanner', spect='SPECT Machine')

        if not machine_type.startswith('http:'):
            self.type = machine_term[machine_type]
        else:    
            self.type = machine_type
        self.prov_type = PROV['Agent']

        if label is None:
            self.label = machine_label[machine_type]
        else:
            self.label = label        

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        # TODO: handle multiple basis
        query = """
        prefix nlx_Imaginginstrument: <http://uri.neuinfo.org/nif/nifstd/birnlex_2094>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nlx_Imaginginstrument: ;
                rdfs:label ?label ;
                rdf:type ?machine_type .
        }
        """
        return query

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['type'], NLX_IMAGING_INSTRUMENT),
            (PROV['label'], self.label)))


class Group(NIDMObject):
    """
    Object representing a ImagingInstrument entity.
    """

    def __init__(self, num_subjects, group_name, label=None, oid=None):
        super(Group, self).__init__(oid=oid)
        self.type = STATO_GROUP
        self.prov_type = PROV['Agent']
        self.group_name = group_name
        self.num_subjects = num_subjects
        if not label:
            label = "Study group population: " + group_name
        self.label = label

    @classmethod
    def get_query(klass, oid=None):
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
        return query

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_GROUP_NAME, self.group_name),
            (NIDM_NUMBER_OF_SUBJECTS, self.num_subjects),
            (PROV['label'], self.label)))


class Person(NIDMObject):
    """
    Object representing a ImagingInstrument entity.
    """

    def __init__(self,  label=None, oid=None):
        super(Person, self).__init__(oid=oid)
        self.prov_type = PROV['Agent']
        self.type = PROV['Person']
        if not label:
            label = "Person"
        self.label = label

    @classmethod
    def get_query(klass, oid=None):
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
        return query

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.prov_type),
            (PROV['type'], self.type),
            (PROV['label'], self.label)))


class DesignMatrix(NIDMObject):

    """
    Object representing a DesignMatrix entity.
    """

    def __init__(self, matrix, image_file, export_dir, regressors=None,
                 design_type=None, hrf_model=None, drift_model=None,
                 suffix='', csv_file=None, filename=None, label=None, oid=None):
        super(DesignMatrix, self).__init__(export_dir=export_dir, oid=oid)
        self.type = NIDM_DESIGN_MATRIX
        self.prov_type = PROV['Entity']
        self.id = NIIRI[str(uuid.uuid4())]
        img_filename = 'DesignMatrix' + suffix + '.png'
        self.image = Image(export_dir, image_file, img_filename)
        self.regressors = regressors
        self.design_type = design_type
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        if csv_file is None:
            self.csv_file = 'DesignMatrix' + suffix + '.csv'
            self.matrix = matrix
        else:
            self.csv_file = csv_file
            # TODO: this fails as csv_file is only a relative path and we don't have the root to append...
            # self.matrix = genfromtxt(self.csv_file, delimiter=',')
            self.matrix = []
        if filename is None:
            self.filename = 'DesignMatrix' + suffix + '.csv'
        else:
            self.filename = filename
        if label is not None:
            self.label = label
        else:
            self.label="Design Matrix"

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        # TODO: handle multiple basis
        query = """
        prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_0000056>
        prefix nidm_withEstimationMethod: <http://purl.org/nidash/nidm#NIDM_0000134>
        prefix nidm_hasHRFBasis: <http://purl.org/nidash/nidm#NIDM_0000102>
        prefix nidm_hasDriftModel: <http://purl.org/nidash/nidm#NIDM_0000088>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_DesignMatrix: ;
                rdfs:label ?label ;
                prov:atLocation ?csv_file ;
                nfo:fileName ?filename .

            OPTIONAL { """  + oid_var + """ nidm_regressorNames: ?regressors . } .
            OPTIONAL { """  + oid_var + """ nidm_hasHRFBasis: ?hrf_model . } .
            OPTIONAL { """  + oid_var + """ nidm_hasDriftModel: ?drift_model . } .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        # Create cvs file containing design matrix
        print(os.path.join(export_dir, self.csv_file))
        print('oooo')
        np.savetxt(os.path.join(export_dir, self.csv_file),
                   np.asarray(self.matrix), delimiter=",")

        if nidm_version['num'] in ["1.0.0", "1.1.0"]:
            csv_location = Identifier("file://./" + self.csv_file)
        else:
            csv_location = Identifier(self.csv_file)

        attributes = [(PROV['type'], self.type),
                      (PROV['label'], self.label),
                      (NIDM_REGRESSOR_NAMES, json.dumps(self.regressors)),
                      (DCT['format'], "text/csv"),
                      (NFO['fileName'], self.filename),
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
            attributes.append((NIDM_HAS_DRIFT_MODEL, self.drift_model.id))

        # Create "design matrix" entity
        self.add_attributes(attributes)


class DriftModel(NIDMObject):

    """
    Object representing a DriftModel entity.
    """

    def __init__(self, drift_type, parameter, oid=None):
        super(DriftModel, self).__init__(oid=oid)
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


class Data(NIDMObject):

    """
    Object representing a Data entity.
    """

    def __init__(self, grand_mean_scaling, target=None, mri_protocol=None,
                 label=None, group_or_sub=None, oid=None):
        super(Data, self).__init__(oid=oid)
        self.grand_mean_sc = grand_mean_scaling
        self.target_intensity = target
        self.type = NIDM_DATA
        self.prov_type = PROV['Entity']
        self.mri_protocol = mri_protocol
        if self.mri_protocol == "fmri":
            self.mri_protocol = NLX_FMRI_PROTOCOL
        if label is None:
            label = "Data"
        self.label = label
        self.group_or_sub = group_or_sub

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """  
        prefix nidm_Data: <http://purl.org/nidash/nidm#NIDM_0000169>
        prefix nidm_grandMeanScaling: <http://purl.org/nidash/nidm#NIDM_0000096>
        prefix nidm_targetIntensity: <http://purl.org/nidash/nidm#NIDM_0000124>
        prefix nidm_hasMRIProtocol: <http://purl.org/nidash/nidm#NIDM_0000172>
        prefix nlx_FunctionalMRIprotocol: <http://uri.neuinfo.org/nif/nifstd/birnlex_2250>


        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_Data: ;
                rdfs:label ?label ;
                nidm_grandMeanScaling: $grand_mean_scaling .
            OPTIONAL {""" + oid_var + """ nidm_targetIntensity: ?target . } .
            OPTIONAL {""" + oid_var + """ nidm_hasMRIProtocol: ?mri_protocol . } .
        }
        """
        return query

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
            (PROV['label'], self.label),
            (NIDM_GRAND_MEAN_SCALING, self.grand_mean_sc),
            (NIDM_TARGET_INTENSITY, self.target_intensity)))

        if nidm_version['major'] > 1 or \
                (nidm_version['major'] == 1 and nidm_version['minor'] > 2):
            if self.mri_protocol is not None:
                self.add_attributes(
                    [(NIDM_HAS_MRI_PROTOCOL, self.mri_protocol)])


class ErrorModel(NIDMObject):

    """
    Object representing an ErrorModel entity.
    """

    def __init__(self, error_distribution, variance_homo, variance_spatial,
                 dependance, dependance_spatial=None, oid=None):
        super(ErrorModel, self).__init__(oid=oid)
        self.error_distribution = error_distribution
        self.variance_homo = variance_homo
        self.variance_spatial = variance_spatial
        self.dependance = dependance
        self.dependance_spatial = dependance_spatial
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = NIDM_ERROR_MODEL
        self.prov_type = PROV['Entity']

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """  
        prefix nidm_ErrorModel: <http://purl.org/nidash/nidm#NIDM_0000023>
        prefix nidm_hasErrorDistribution: <http://purl.org/nidash/nidm#NIDM_0000101>
        prefix nidm_errorVarianceHomogeneous: <http://purl.org/nidash/nidm#NIDM_0000094>
        prefix nidm_varianceMapWiseDependence: <http://purl.org/nidash/nidm#NIDM_0000126>
        prefix nidm_hasErrorDependence: <http://purl.org/nidash/nidm#NIDM_0000100>
        prefix nidm_dependenceMapWiseDependence: <http://purl.org/nidash/nidm#NIDM_0000089>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ErrorModel: ;
                rdfs:label ?label ;
                nidm_hasErrorDistribution: $error_distribution ;
                nidm_errorVarianceHomogeneous: $variance_homo ;
                nidm_varianceMapWiseDependence: $variance_spatial ;
                nidm_hasErrorDependence: $dependance .

            OPTIONAL {""" + oid_var + """ nidm_dependenceMapWiseDependence: ?dependance_spatial . } .
        }
        """
        return query

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


class ModelParametersEstimation(NIDMObject):

    """
    Object representing an ModelParametersEstimation activity.
    """

    def __init__(self, estimation_method, software_id, data=None, label=None,
                 oid=None):
        super(ModelParametersEstimation, self).__init__(oid=oid)
        self.estimation_method = estimation_method
        self.software_id = software_id
        self.type = NIDM_MODEL_PARAMETERS_ESTIMATION
        self.prov_type = PROV['Activity']
        # currenlty only used for reading
        self.data = data
        if label is None:
            label = "Model Parameters Estimation"
        self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_0000056>
        prefix nidm_withEstimationMethod: <http://purl.org/nidash/nidm#NIDM_0000134>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ModelParameterEstimation: ;
                rdfs:label ?label ;
                nidm_withEstimationMethod: ?estimation_method .
        }
        """
        return query

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        # Create "Model Parameter estimation" activity
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_WITH_ESTIMATION_METHOD, self.estimation_method),
            (PROV['label'], self.label)))


class ParameterEstimateMap(NIDMObject):

    """
    Object representing an ParameterEstimateMap entity.
    """

    def __init__(self, pe_file, pe_num, coord_space, filename=None, sha=None,
                 label=None, suffix='', model_param_estimation=None, oid=None,
                 export_dir=None, format=None):
        super(ParameterEstimateMap, self).__init__(oid=oid)
        # Column index in the corresponding design matrix
        self.num = pe_num
        self.coord_space = coord_space
        # Parameter Estimate Map is going to be copied over to export_dir
        if export_dir is not None:
            filename = 'ParameterEstimate' + suffix + '.nii.gz'
        else:
            filename = filename

        self.file = NIDMFile(self.id, pe_file, new_filename=filename, sha=sha,
                             export_dir=export_dir, format=format)

        self.type = NIDM_PARAMETER_ESTIMATE_MAP
        self.prov_type = PROV['Entity']
        if label is None:
            label = "Parameter estimate " + str(self.num)
        self.label = label
        # Only used for reading (so far)
        self.model_param_estimation = model_param_estimation

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ParameterEstimateMap: <http://purl.org/nidash/nidm#NIDM_0000061>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ParameterEstimateMap: ;
                rdfs:label ?label ;
                nfo:fileName ?filename ;
                crypto:sha512 ?sha ;
                prov:atLocation ?pe_file ;
                dct:format ?format .
        }
        """
        return query

    # Generate prov for contrast map
    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        # Parameter estimate entity
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], self.label)))


class ResidualMeanSquares(NIDMObject):

    """
    Object representing an ResidualMeanSquares entity.
    """

    def __init__(self, export_dir, residual_file, coord_space,
                 temporary=False, suffix='', format=None, filename=None,
                 sha=None, label=None, oid=None):
        super(ResidualMeanSquares, self).__init__(export_dir, oid=oid)
        self.coord_space = coord_space
        self.id = NIIRI[str(uuid.uuid4())]
        if filename is None:
            filename = 'ResidualMeanSquares' + suffix + '.nii.gz'
        self.file = NIDMFile(self.id, residual_file, filename, export_dir,
                             temporary=temporary, format=format, sha=sha)
        if label is None:
            label = "Residual Mean Squares Map"
        self.label = label
        self.type = NIDM_RESIDUAL_MEAN_SQUARES_MAP
        self.prov_type = PROV['Entity']

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ResidualMeanSquaresMap: <http://purl.org/nidash/nidm#NIDM_0000066>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ResidualMeanSquaresMap: ;
                rdfs:label ?label ;
                nfo:fileName ?filename ;
                crypto:sha512 ?sha ;
                prov:atLocation ?residual_file ;
                dct:format ?format .
        }
        """
        return query

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type,),
            (PROV['label'], self.label),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id)))


class MaskMap(NIDMObject):

    """
    Object representing an MaskMap entity.
    """

    def __init__(self, export_dir, mask_file, coord_space, user_defined,
                 suffix='', filename=None, format=None, label=None, sha=None, oid=None):
        super(MaskMap, self).__init__(export_dir, oid=oid)
        self.coord_space = coord_space
        self.id = NIIRI[str(uuid.uuid4())]
        if filename is None:
            filename = 'Mask' + suffix + '.nii.gz'
        self.file = NIDMFile(self.id, mask_file, filename, export_dir, 
            sha=sha, format=format)
        self.user_defined = user_defined
        self.type = NIDM_MASK_MAP
        self.prov_type = PROV['Entity']
        if label is None:
            label = "Mask"
        self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_MaskMap: <http://purl.org/nidash/nidm#NIDM_0000054>
        prefix nidm_isUserDefined: <http://purl.org/nidash/nidm#NIDM_0000106>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_MaskMap: ;
                rdfs:label ?label ;
                nidm_isUserDefined: ?user_defined ;
                nfo:fileName ?filename ;
                crypto:sha512 ?sha ;
                prov:atLocation ?mask_file ;
                dct:format ?format .
        }
        """
        return query

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type,),
            (PROV['label'], self.label),
            (NIDM_IS_USER_DEFINED, self.user_defined),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id))
        )


class GrandMeanMap(NIDMObject):

    """
    Object representing an GrandMeanMap entity.
    """

    # TODO: we should remove mask_file here and ask for masked data instead?
    def __init__(self, org_file, mask_file, coord_space, export_dir,
                 suffix='', label=None, filename=None, sha=None,
                 format=format, masked_median=None, oid=None):
        super(GrandMeanMap, self).__init__(export_dir, oid=oid)
        self.id = NIIRI[str(uuid.uuid4())]
        if filename is None:
            filename = 'GrandMean' + suffix + '.nii.gz'
        self.file = NIDMFile(self.id, org_file, filename, export_dir, 
            sha=sha, format=format)
        self.mask_file = mask_file  # needed to compute masked median
        self.coord_space = coord_space
        self.type = NIDM_GRAND_MEAN_MAP
        self.prov_type = PROV['Entity']
        if label is None:
            label = "Grand Mean Map"
        self.label = label
        self.masked_median = masked_median

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_GrandMeanMap: <http://purl.org/nidash/nidm#NIDM_0000033>
        prefix nidm_maskedMedian: <http://purl.org/nidash/nidm#NIDM_0000107>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_GrandMeanMap: ;
                rdfs:label ?label ;
                nidm_maskedMedian: ?masked_median;
                prov:atLocation ?org_file ;
                nfo:fileName ?filename ;
                crypto:sha512 ?sha ;
                dct:format ?format .
        }
        """
        return query

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """

        if self.masked_median is None:
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
            (PROV['label'], self.label),
            (NIDM_MASKED_MEDIAN, masked_median),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id))
        )
