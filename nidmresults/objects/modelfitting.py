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


class ModelFitting(NIDMObject):

    """
    Object representing a Model fitting step: including a
    ModelParametersEstimation activity, its inputs and outputs.
    """

    def __init__(self, activity, design_matrix, data, error_model,
                 param_estimates, rms_map, mask_map, grand_mean_map,
                 machine, subjects, rpv_map=None):
        self.activity = activity
        self.design_matrix = design_matrix
        self.data = data
        self.error_model = error_model
        self.param_estimates = param_estimates
        self.rms_map = rms_map
        self.rpv_map = rpv_map
        self.mask_map = mask_map
        self.grand_mean_map = grand_mean_map
        self.machine = machine
        self.subjects = subjects

        # Useful for printing
        self.label = 'Model fitting'

    @classmethod
    def get_query(klass):
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
        return query

    @classmethod
    def load_from_json(klass, json_dict, base_dir, software_id):
        # TODO: currently assuming list of 1 ==> should be extended
        model_fittings = list()

        activity = ModelParametersEstimation.load(json_dict, software_id)
        design = DesignMatrix.load(json_dict)
        data = Data.load(json_dict)
        error = ErrorModel.load(json_dict)
        param_estimates = ParameterEstimateMap.load(json_dict, base_dir)
        rms_map = ResidualMeanSquares.load(json_dict, base_dir)
        mask_map = MaskMap.load(json_dict, base_dir)
        grand_mean_map = GrandMeanMap.load(json_dict, base_dir)
        machine = ImagingInstrument.load(json_dict)
        subjects = Group.load(json_dict)

        mf = ModelFitting(
                activity, design, data, error,
                param_estimates, rms_map, mask_map, grand_mean_map,
                machine, subjects, rpv_map=None)

        return mf


class ImagingInstrument(NIDMObject):
    """
    Object representing a ImagingInstrument entity.
    """

    def __init__(self, machine_type, label=None, oid=None):
        super(ImagingInstrument, self).__init__(oid=oid)

        machine_label = dict()
        machine_label[NIF_MRI] = 'MRI Scanner'
        machine_label[NIF_EEG] = 'EEG Machine'
        machine_label[NIF_MEG] = 'MEG Machine'
        machine_label[NIF_PET] = 'PET Scanner'
        machine_label[NIF_SPECT] = 'SPECT Machine'

        if not isinstance(machine_type, QualifiedName):
            machine_type = machine_type.lower()
            machine_term = dict(
                mri=NIF_MRI, eeg=NIF_EEG, meg=NIF_MEG, pet=NIF_PET,
                spect=NIF_SPECT)

            if not machine_type.startswith('http:'):
                self.type = machine_term[machine_type]
        else:
            self.type = machine_type

        self.prov_type = PROV['Agent']

        if label is None:
            self.label = machine_label[self.type]
        else:
            self.label = label

    @classmethod
    def load_from_json(klass, json_dict):
        MACHINES = {
            'nlx_ElectroencephalographyMachine': NIF_EEG,
            'nlx_MagnetoencephalographyMachine': NIF_MEG,
            'nlx_PositronEmissionTomographyScanner': NIF_PET,
            'nlx_SinglePhotonEmissionComputedTomographyScanner': NIF_SPECT,
            'nlx_MagneticResonanceImagingScanner': NIF_MRI
        }
        machine_type = MACHINES[json_dict['ImagingInstrument_type']]
        instrument = ImagingInstrument(machine_type)
        return instrument

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        # TODO: handle multiple basis
        query = """
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

SELECT DISTINCT * WHERE {
    {""" + oid_var + """ a nlx_Imaginginstrument: .} UNION
    {""" + oid_var + """ a nlx_MagneticResonanceImagingScanner: .} UNION
    {""" + oid_var + """ a nlx_PositronEmissionTomographyScanner: .} UNION
    {""" + oid_var + """ a nlx_SinglePhotonEmissionComputedTomographyScanner: .} UNION
    {""" + oid_var + """ a nlx_MagnetoencephalographyMachine: .} UNION
    {""" + oid_var + """ a nlx_ElectroencephalographyMachine: .}

    """ + oid_var + """ rdfs:label ?label ;
        rdf:type ?machine_type .

    FILTER ( ?machine_type NOT IN (prov:Agent, prov:SoftwareAgent, nlx_Imaging\
instrument:) )
}
        """
        return query

    def export(self, nidm_version, export_dir):
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

    @classmethod
    def load_from_json(klass, json_dict):
        groups = json_dict.get('groups', None)
        grps = list()

        if groups is not None:
            for group in groups:
                group_name = group['StudyGroupPopulation_groupName']
                num_subjects = group['StudyGroupPopulation_numberOfSubjects']
                grp = Group(num_subjects, group_name)
                grps.append(grp)
        else:
            grps.append(Person())

        return grps

    def export(self, nidm_version, export_dir):
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

    def export(self, nidm_version, export_dir):
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

    def __init__(self, matrix, image_file, regressors=None,
                 design_type=None, hrf_models=None, drift_model=None,
                 suffix='', csv_file=None, filename=None, label=None,
                 oid=None):
        super(DesignMatrix, self).__init__(oid=oid)
        self.type = NIDM_DESIGN_MATRIX
        self.prov_type = PROV['Entity']
        img_filename = 'DesignMatrix' + suffix + '.png'
        if isinstance(image_file, Image):
            self.image = image_file
        else:
            self.image = Image(image_file, img_filename)

        # Note: changed to fit regressors passed as loaded json when creating
        # NIDM pack from JSON --> check if this cause issue in the tests TODO
        if not type(regressors) is not list:
            regressors = json.loads(regressors)
        self.regressors = regressors

        self.design_type = design_type
        self.hrf_models = hrf_models

        self.drift_model = drift_model
        if csv_file is None:
            self.csv_file = 'DesignMatrix' + suffix + '.csv'
            self.matrix = matrix
        else:
            self.csv_file = csv_file
            # TODO: this fails as csv_file is only a relative path and we don't
            # have the root to append...
            # self.matrix = genfromtxt(self.csv_file, delimiter=',')
            self.matrix = []
        if filename is None:
            self.filename = 'DesignMatrix' + suffix + '.csv'
        else:
            self.filename = filename
        if label is not None:
            self.label = label
        else:
            self.label = "Design Matrix"

    @classmethod
    def load_from_json(klass, json_dict):
        if 'DesignMatrix_atLocation' in json_dict:
            mat_csv = json_dict['DesignMatrix_atLocation']
            # Note: this could be removed and the csv passed directly
            matrix = genfromtxt(mat_csv, delimiter=',')
        else:
            matrix = json_dict['DesignMatrix_value']

        # TODO: deal with optional png of design matric
        image_file = None

        regressors = json_dict['DesignMatrix_regressorNames']
        # TODO deal with optional arguments

        design = DesignMatrix(
                    matrix, image_file, regressors=None,
                    design_type=None, hrf_models=None, drift_model=None,
                    suffix='', csv_file=None, filename=None, label=None,
                    oid=None)
        return design

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        # TODO: handle multiple basis
        query = """
prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_000005\
6>
prefix nidm_withEstimationMethod: <http://purl.org/nidash/nidm#NIDM_0000134>
prefix nidm_hasHRFBasis: <http://purl.org/nidash/nidm#NIDM_0000102>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_DesignMatrix: ;
        rdfs:label ?label ;
        prov:atLocation ?csv_file ;
        nfo:fileName ?filename .

    OPTIONAL { """ + oid_var + """ nidm_regressorNames: ?regressors . } .
}
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        # Create cvs file containing design matrix
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

        if self.hrf_models is not None:
            if nidm_version['num'] in ("1.0.0", "1.1.0"):
                if self.design_type is not None:
                    attributes.append(
                        (NIDM_HAS_FMRI_DESIGN, self.design_type))
                else:
                    warnings.warn("Design type is missing")

            # hrf model
            for hrf_model in self.hrf_models:
                attributes.append((NIDM_HAS_HRF_BASIS, hrf_model))
            # drift model
            if self.drift_model is not None:
                attributes.append((NIDM_HAS_DRIFT_MODEL, self.drift_model.id))

        # Create "design matrix" entity
        self.add_attributes(attributes)


class DriftModel(NIDMObject):

    """
    Object representing a DriftModel entity.
    """

    def __init__(self, drift_type, parameter, label=None, oid=None):
        super(DriftModel, self).__init__(oid=oid)

        # if not isinstance(drift_type, QualifiedName):
        #     drift_type = namespace_manager.valid_qualified_name(drift_type)

        self.drift_type = drift_type
        self.parameter = parameter
        self.type = drift_type
        self.prov_type = PROV['Entity']
        if not label:
            if self.drift_type == FSL_GAUSSIAN_RUNNING_LINE_DRIFT_MODEL:
                self.label = "FSL's Gaussian Running Line Drift Model"
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_DesignMatrix: <http://purl.org/nidash/nidm#NIDM_0000019>
prefix spm_SPMsDriftCutoffPeriod: <http://purl.org/nidash/spm#SPM_0000001>
prefix fsl_driftCutoffPeriod: <http://purl.org/nidash/fsl#FSL_0000004>

SELECT DISTINCT * WHERE {
    [] a nidm_DesignMatrix: ;
        nidm_hasDriftModel: """ + oid_var + """ .

    """ + oid_var + """ a ?drift_type ;
        rdfs:label ?label .

    {""" + oid_var + """ spm_SPMsDriftCutoffPeriod: ?parameter .} UNION
    {""" + oid_var + """ fsl_driftCutoffPeriod: ?parameter .} .

    FILTER ( ?drift_type NOT IN (prov:Entity) )
}
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        attributes = [(PROV['type'], self.drift_type),
                      (PROV['label'], self.label)]

        if self.drift_type == FSL_GAUSSIAN_RUNNING_LINE_DRIFT_MODEL:
            attributes.append((FSL_DRIFT_CUTOFF_PERIOD, self.parameter))

        if self.drift_type == SPM_DCT_DRIFT_MODEL:
            attributes.append((SPM_SPMS_DRIFT_CUT_OFF_PERIOD, self.parameter))

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
    def load_from_json(klass, json_dict):
        grand_mean_scaling = json_dict['Data_grandMeanScaling']
        target = json_dict['Data_targetIntensity']
        # TODO deal with optional arguments
        data = Data(grand_mean_scaling, mri_protocol=None,
                 label=None, group_or_sub=None, oid=None)
        return data

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
prefix nlx_FunctionalMRIprotocol: <http://uri.neuinfo.org/nif/nifstd/birnlex_2\
250>


SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_Data: ;
        rdfs:label ?label ;
        nidm_grandMeanScaling: $grand_mean_scaling .
    OPTIONAL {""" + oid_var + """ nidm_targetIntensity: ?target . } .
    OPTIONAL {""" + oid_var + """ nidm_hasMRIProtocol: ?mri_protocol . } .
}
        """
        return query

    def export(self, nidm_version, export_dir):
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
        self.type = NIDM_ERROR_MODEL
        self.prov_type = PROV['Entity']

    @classmethod
    def load_from_json(klass, json_dict):
        DEP = {
            'nidm_ConstantParameter': SPATIALLY_GLOBAL,
            'nidm_IndependentParameter': SPATIALLY_LOCAL,
            'nidm_RegularizedParameter': SPATIALLY_REGUL,
        }
        DIST = {
            'obo_NormalDistribution': STATO_NORMAL_DISTRIBUTION,
        }

        error_distribution = DIST[json_dict['ErrorModel_hasErrorDistribution']]
        variance_homo = json_dict['ErrorModel_errorVarianceHomogeneous']
        variance_spatial = DEP[
            json_dict['ErrorModel_varianceMapWiseDependence']]
        dep = json_dict['ErrorModel_hasErrorDependence']
        dep_spatial = DEP[json_dict['ErrorModel_dependenceMapWiseDependence']]

        error = ErrorModel(
                    error_distribution, variance_homo, variance_spatial,
                    dep, dep_spatial, oid=None)
        return error

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_ErrorModel: <http://purl.org/nidash/nidm#NIDM_0000023>
prefix nidm_hasErrorDistribution: <http://purl.org/nidash/nidm#NIDM_0000101>
prefix nidm_errorVarianceHomogeneous: <http://purl.org/nidash/nidm#NIDM_000009\
4>
prefix nidm_varianceMapWiseDependence: <http://purl.org/nidash/nidm#NIDM_00001\
26>
prefix nidm_hasErrorDependence: <http://purl.org/nidash/nidm#NIDM_0000100>
prefix nidm_dependenceMapWiseDependence: <http://purl.org/nidash/nidm#NIDM_000\
0089>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_ErrorModel: ;
        nidm_hasErrorDistribution: $error_distribution ;
        nidm_errorVarianceHomogeneous: $variance_homo ;
        nidm_varianceMapWiseDependence: $variance_spatial ;
        nidm_hasErrorDependence: $dependance .

    OPTIONAL {""" + oid_var + """ rdfs:label ?label . } .
    OPTIONAL {""" + oid_var + """ nidm_dependenceMapWiseDependence: ?dependance_spatial . } .
}
        """
        return query

    def export(self, nidm_version, export_dir):
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
prefix nidm_ModelParameterEstimation: <http://purl.org/nidash/nidm#NIDM_000005\
6>
prefix nidm_withEstimationMethod: <http://purl.org/nidash/nidm#NIDM_0000134>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_ModelParameterEstimation: ;
        rdfs:label ?label ;
        nidm_withEstimationMethod: ?estimation_method ;
        prov:wasAssociatedWith ?software_id .
}
        """
        return query

    @classmethod
    def load_from_json(self, json_dict, software_id):
        est_method = json_dict['ModelParameterEstimation_withEstimationMethod']
        return ModelParametersEstimation(est_method, software_id)

    def export(self, nidm_version, export_dir):
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

    def __init__(self, coord_space=None, pe_file=None, pe_num=None, filename=None,
                 sha=None, label=None, suffix='', model_param_estimation=None,
                 oid=None, fmt=None, derfrom_id=None, derfrom_filename=None,
                 derfrom_fmt=None, derfrom_sha=None, isderfrommap=False):
        super(ParameterEstimateMap, self).__init__(oid=oid)
        # Column index in the corresponding design matrix
        self.num = pe_num

        self.coord_space = coord_space

        # Parameter Estimate Map is going to be copied over to export_dir
        if not filename:
            if suffix is None and pe_num is not None:
                suffix = str(pe_num)
            filename = 'ParameterEstimate' + suffix + '.nii.gz'

        self.file = NIDMFile(self.id, pe_file, filename=filename, sha=sha,
                             fmt=fmt)

        self.type = NIDM_PARAMETER_ESTIMATE_MAP
        self.prov_type = PROV['Entity']
        if label is None:
            if self.num:
                label = "Parameter estimate " + str(self.num)
            else:
                label = None

        self.label = label
        # Only used for reading (so far)
        self.model_param_estimation = model_param_estimation

        if derfrom_id is not None:
            self.derfrom = ParameterEstimateMap(
                oid=derfrom_id, coord_space=coord_space,
                filename=derfrom_filename, sha=derfrom_sha,
                fmt=derfrom_fmt, isderfrommap=True)
        else:
            self.derfrom = None
        self.isderfrommap = isderfrommap

    @classmethod
    def load_from_json(klass, json_dict, base_dir):
        pe_list = list()
        params = json_dict['ParameterEstimateMaps']

        for idx, pe_file in enumerate(params):
            # FIXME: deal with varying coordsys across maps
            coordspace = CoordinateSpace.load_from_json(json_dict, 
                os.path.join(base_dir, pe_file))

            pe = ParameterEstimateMap(coordspace, pe_file, idx+1)
            pe_list.append(pe)
        
        return pe_list

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
        dct:format ?fmt .

    OPTIONAL {""" + oid_var + """ prov:atLocation ?pe_file .} .

    OPTIONAL {""" + oid_var + """ prov:wasDerivedFrom ?derfrom_id .

    ?derfrom_id a nidm_ParameterEstimateMap: ;
        nfo:fileName ?derfrom_filename ;
        dct:format ?derfrom_fmt ;
        crypto:sha512 ?derfrom_sha .
     } .
}
        """

#     ?derfrom_id a nidm_ParameterEstimateMap: ;
# nfo:fileName ?derfrom_filename ;
# dct:format ?derfrom_format ;
# crypto:sha512 ?derfrom_sha . } .

        return query

    # Generate prov for contrast map
    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        atts = (
            (PROV['type'], self.type),)

        if not self.isderfrommap:
            atts = atts + (
                (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),)

        if self.label is not None:
            atts = atts + (
                (PROV['label'], self.label),)

        # Parameter estimate entity
        self.add_attributes(atts)


class ResidualMeanSquares(NIDMObject):

    """
    Object representing an ResidualMeanSquares entity.
    """

    def __init__(self, residual_file, coord_space,
                 temporary=False, suffix='', fmt=None, filename=None,
                 sha=None, label=None, oid=None,
                 derfrom_id=None, derfrom_filename=None, derfrom_fmt=None,
                 derfrom_sha=None, isderfrommap=False):
        super(ResidualMeanSquares, self).__init__(oid=oid)
        self.coord_space = coord_space
        if filename is None:
            filename = 'ResidualMeanSquares' + suffix + '.nii.gz'
        self.file = NIDMFile(self.id, residual_file, filename,
                             temporary=temporary, fmt=fmt, sha=sha)
        if label is None:
            label = "Residual Mean Squares Map"
        self.label = label
        self.type = NIDM_RESIDUAL_MEAN_SQUARES_MAP
        self.prov_type = PROV['Entity']
        if derfrom_id is not None:
            self.derfrom = ResidualMeanSquares(
                None, coord_space,
                oid=derfrom_id, filename=derfrom_filename,
                sha=derfrom_sha, fmt=derfrom_fmt,
                isderfrommap=True)
        else:
            self.derfrom = None
        self.isderfrommap = isderfrommap

    @classmethod
    def load_from_json(klass, json_dict, base_dir):
        rms_file = json_dict['ResidualMeanSquaresMap_atLocation']
        # FIXME: deal with varying coordsys across maps
        coordspace = CoordinateSpace.load_from_json(json_dict, 
            os.path.join(base_dir, rms_file))
        rms = ResidualMeanSquares(rms_file, coordspace)
       
        return rms

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
        dct:format ?fmt .

    OPTIONAL {""" + oid_var + """ prov:wasDerivedFrom ?derfrom_id .

    ?derfrom_id a nidm_ResidualMeanSquaresMap: ;
        nfo:fileName ?derfrom_filename ;
        dct:format ?derfrom_fmt ;
        crypto:sha512 ?derfrom_sha .
     } .

}
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        atts = (
            (PROV['type'], self.type,),
            )

        if not self.isderfrommap:
            atts = atts + (
                (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
                (PROV['label'], self.label))

        self.add_attributes(atts)


class ReselsPerVoxelMap(NIDMObject):

    """
    Object representing an ResidualMeanSquares entity.
    """

    def __init__(self, rpv_file, coord_space,
                 temporary=False, suffix='', fmt=None, filename=None,
                 sha=None, label=None, oid=None,
                 derfrom_id=None, derfrom_filename=None, derfrom_fmt=None,
                 derfrom_sha=None, inf_id=None, isderfrommap=False):
        super(ReselsPerVoxelMap, self).__init__(oid=oid)
        self.coord_space = coord_space
        if filename is None:
            filename = 'ReselsPerVoxelMap' + suffix + '.nii.gz'
        self.file = NIDMFile(self.id, rpv_file, filename,
                             temporary=temporary, fmt=fmt, sha=sha)
        if label is None:
            label = "Resels Per Voxel File"
        self.label = label
        self.type = NIDM_RESELS_PER_VOXEL_MAP
        self.prov_type = PROV['Entity']
        if derfrom_id is not None:
            self.derfrom = ReselsPerVoxelMap(
                None, coord_space,
                oid=derfrom_id, filename=derfrom_filename,
                sha=derfrom_sha, fmt=derfrom_fmt,
                isderfrommap=True)
        else:
            self.derfrom = None
        self.inf_id = inf_id
        self.isderfrommap = isderfrommap

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_ReselsPerVoxelMap: <http://purl.org/nidash/nidm#NIDM_0000144>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_ReselsPerVoxelMap: ;
        rdfs:label ?label ;
        nfo:fileName ?filename ;
        crypto:sha512 ?sha ;
        prov:atLocation ?rpv_file ;
        dct:format ?fmt .

    ?inf_id prov:used """ + oid_var + """ .

    OPTIONAL {""" + oid_var + """ prov:wasDerivedFrom ?derfrom_id .

    ?derfrom_id a nidm_ReselsPerVoxelMap: ;
        nfo:fileName ?derfrom_filename ;
        dct:format ?derfrom_fmt ;
        crypto:sha512 ?derfrom_sha .
     } .
}
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        atts = (
            (PROV['type'], self.type,),
            )

        if not self.isderfrommap:
            atts = atts + (
                (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
                (PROV['label'], self.label))

        self.add_attributes(atts)


class MaskMap(NIDMObject):

    """
    Object representing an MaskMap entity.
    """

    def __init__(self, mask_file, coord_space, user_defined,
                 suffix='', filename=None, fmt=None, label=None, sha=None,
                 oid=None,
                 derfrom_id=None, derfrom_filename=None, derfrom_fmt=None,
                 derfrom_sha=None, isderfrommap=False):
        super(MaskMap, self).__init__(oid=oid)
        self.coord_space = coord_space
        if filename is None:
            filename = 'Mask' + suffix + '.nii.gz'
        self.file = NIDMFile(self.id, mask_file, filename,
                             sha=sha, fmt=fmt)
        self.user_defined = user_defined
        self.type = NIDM_MASK_MAP
        self.prov_type = PROV['Entity']
        if label is None:
            label = "Mask"
        self.label = label
        if derfrom_id is not None:
            self.derfrom = MaskMap(
                None, coord_space, user_defined,
                oid=derfrom_id, filename=derfrom_filename,
                sha=derfrom_sha, fmt=derfrom_fmt,
                isderfrommap=True)
        else:
            self.derfrom = None
        self.isderfrommap = isderfrommap

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
                dct:format ?fmt .

            OPTIONAL {""" + oid_var + """ prov:wasDerivedFrom ?derfrom_id .

            ?derfrom_id a nidm_MaskMap: ;
                nfo:fileName ?derfrom_filename ;
                dct:format ?derfrom_fmt ;
                crypto:sha512 ?derfrom_sha .
             } .
        }
        """
        return query

    @classmethod
    def load_from_json(klass, json_dict, base_dir):
        mask_file = json_dict['MaskMap_atLocation']
        # FIXME: deal with varying coordsys across maps
        coordspace = CoordinateSpace.load_from_json(json_dict, 
            os.path.join(base_dir, mask_file))
        mask = MaskMap(mask_file, coordspace, False)
       
        return mask


    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        atts = (
            (PROV['type'], self.type,),
        )

        if not self.isderfrommap:
            atts = atts + (
                (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
                (PROV['label'], self.label),
                (NIDM_IS_USER_DEFINED, self.user_defined))

        self.add_attributes(atts)


class GrandMeanMap(NIDMObject):

    """
    Object representing an GrandMeanMap entity.
    """

    # TODO: we should remove mask_file here and ask for masked data instead?
    def __init__(self, org_file, mask_file, coord_space,
                 suffix='', label=None, filename=None, sha=None,
                 fmt=None, masked_median=None, oid=None):
        super(GrandMeanMap, self).__init__(oid=oid)
        if filename is None:
            filename = 'GrandMean' + suffix + '.nii.gz'
        self.file = NIDMFile(self.id, org_file, filename,
                             sha=sha, fmt=fmt)
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
                dct:format ?fmt .
        }
        """
        return query

    @classmethod
    def load_from_json(klass, json_dict, base_dir):
        gm_file = os.path.join(base_dir, json_dict['GrandMeanMap_atLocation'])
        # FIXME: deal with varying coordsys across maps
        coordspace = CoordinateSpace.load_from_json(json_dict, gm_file)
        mask = GrandMeanMap(gm_file, gm_file, coordspace)

        return mask

    def export(self, nidm_version, export_dir):
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
            self.masked_median = np.median(
                np.array(grand_mean_data_in_mask, dtype=float))

        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['label'], self.label),
            (NIDM_MASKED_MEDIAN, self.masked_median),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id))
        )
