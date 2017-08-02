"""
Objects describing the Contrast Estimation activity, its inputs and outputs as
specified in NIDM-Results.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""
import numpy as np
import os
from nidmresults.objects.constants import *
import nibabel as nib
from nidmresults.objects.generic import *
import uuid
from prov.model import Identifier


class Contrast(object):

    """
    Object representing a Contrast Estimation step: including a
    ContrastEstimation activity, its inputs and outputs.
    """

    def __init__(self, contrast_num, contrast_name, weights, estimation,
                 contrast_map, stderr_or_expl_mean_sq_map, stat_map,
                 z_stat_map=None):
        super(Contrast, self).__init__()
        # FIXME: contrast_num migth only be defined in FSL if this is not
        # generic the class should be overloaded in fsl_objects
        self.contrast_num = contrast_num
        self.contrast_name = contrast_name
        self.weights = weights
        self.estimation = estimation
        self.contrast_map = contrast_map
        self.stderr_or_expl_mean_sq_map = stderr_or_expl_mean_sq_map
        self.stat_map = stat_map
        self.z_stat_map = z_stat_map


class ContrastWeights(NIDMObject):

    """
    Object representing a ContrastWeight entity.
    """

    def __init__(self, contrast_num, contrast_name, contrast_weights,
                 stat_type, label=None, oid=None):
        super(ContrastWeights, self).__init__(oid=oid)
        self.contrast_name = contrast_name
        self.contrast_weights = contrast_weights
        self.contrast_num = contrast_num
        self.stat_type = stat_type
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = STATO_CONTRAST_WEIGHT_MATRIX
        self.prov_type = PROV['Entity']
        if label is None:
            self.label = "Contrast Weights: " + self.contrast_name,
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix obo_contrastweightmatrix: <http://purl.obolibrary.org/obo/STATO_0000323>
        prefix nidm_statisticType: <http://purl.org/nidash/nidm#NIDM_0000123>
        prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a obo_contrastweightmatrix: ;
            rdfs:label ?label ;
            prov:value ?contrast_weights ;
            nidm_statisticType: ?stat_type ; 
            nidm_contrastName: ?contrast_name .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """
        if self.stat_type.lower() == "t":
            stat = STATO_TSTATISTIC
        elif self.stat_type.lower() == "z":
            stat = STATO_ZSTATISTIC
        elif self.stat_type.lower() == "f":
            stat = STATO_FSTATISTIC
        elif self.stat_type.startswith('http'):
            stat = Identifier(self.stat_type)

        self.add_attributes((
            (PROV['type'], STATO_CONTRAST_WEIGHT_MATRIX),
            (NIDM_STATISTIC_TYPE, stat),
            (PROV['label'], self.label),
            (NIDM_CONTRAST_NAME, self.contrast_name),
            (PROV['value'], self.contrast_weights)))


class ContrastMap(NIDMObject):

    """
    Object representing a ContrastMap entity.
    """

    def __init__(self, contrast_file, contrast_num, contrast_name,
                 coord_space, ident=None, sha=None, format=None, 
                 label=None, filename=None, oid=None):
        super(ContrastMap, self).__init__(oid=oid)
        self.num = contrast_num
        self.name = contrast_name
        if ident is None:
            self.id = NIIRI[str(uuid.uuid4())]
        else:
            self.id = ident
        if filename is None:
            filename = 'Contrast' + self.num + '.nii.gz'
        self.file = NIDMFile(self.id, contrast_file, filename, sha=sha, format=format)
        self.coord_space = coord_space
        self.type = NIDM_CONTRAST_MAP
        self.prov_type = PROV['Entity']
        if label is None:
            self.label = "Contrast Map: " + self.name
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ContrastMap: <http://purl.org/nidash/nidm#NIDM_0000002>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ContrastMap: ;
            rdfs:label ?label ;
            prov:atLocation ?contrast_file ;
            dct:format ?format ;
            nfo:fileName ?filename ;
            nidm_contrastName: ?contrast_name ;
            crypto:sha512 ?sha ;
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """
        # Contrast Map entity
        self.add_attributes((
            (PROV['type'], NIDM_CONTRAST_MAP),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (NIDM_CONTRAST_NAME, self.name),
            (PROV['label'], self.label)))


class ContrastExplainedMeanSquareMap(NIDMObject):
    """
    Object representing a ContrastExplainedMeanSquareMap entity.
    """
    def __init__(self, stat_file, sigma_sq_file, contrast_num,
                 coord_space, expl_mean_sq_file=None, 
                 sha=None, format=None, filename=None, oid=None):
        super(ContrastExplainedMeanSquareMap, self).__init__(oid=oid)
        self.stat_file = stat_file
        self.sigma_sq_file = sigma_sq_file
        self.num = contrast_num
        self.id = NIIRI[str(uuid.uuid4())]
        self.coord_space = coord_space
        self.type = NIDM_CONTRAST_EXPLAINED_MEAN_SQUARE_MAP
        self.prov_type = PROV['Entity']
        self.expl_mean_sq_file = expl_mean_sq_file
        self.sha = sha
        self.filename = filename
        self.format = format


    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ContrastExplainedMeanSquareMap: <http://purl.org/nidash/nidm#NIDM_0000163>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ContrastExplainedMeanSquareMap: ;
            rdfs:label ?label ;
            prov:atLocation ?contrast_file ;
            dct:format ?format ;
            nfo:fileName ?filename ;
            crypto:sha512 ?sha .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """
        if self.expl_mean_sq_file is None:
            # Create Contrast Explained Mean Square Map as fstat<num>.nii.gz
            # multiplied by sigmasquareds.nii.gz and save it in export_dir
            fstat_img = nib.load(self.stat_file)
            fstat = fstat_img.get_data()

            sigma_sq_img = nib.load(self.sigma_sq_file)
            sigma_sq = sigma_sq_img.get_data()

            expl_mean_sq = nib.Nifti1Image(fstat*sigma_sq, fstat_img.get_qform())

            expl_mean_sq_filename = \
                "ContrastExplainedMeanSquareMap" + self.num + ".nii.gz"
            self.expl_mean_sq_file = os.path.join(export_dir, expl_mean_sq_filename)
            nib.save(expl_mean_sq, expl_mean_sq_file)

        self.file = NIDMFile(self.id, expl_mean_sq_file,
                             expl_mean_sq_filename,
                             sha=self.sha, filename=self.filename, format=self.format)

        # Contrast Explained Mean Square Map entity
        path, filename = os.path.split(expl_mean_sq_file)
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], "Contrast Explained Mean Square Map")))


class ContrastStdErrMap(NIDMObject):

    """
    Object representing a ContrastStdErrMap entity.
    """

    def __init__(self, contrast_num, filepath, is_variance, coord_space,
                 var_coord_space, label=None, format=None, 
                 sha=None, filename=None, oid=None):
        super(ContrastStdErrMap, self).__init__(oid=oid)
        self.file = filepath
        self.id = NIIRI[str(uuid.uuid4())]
        self.is_variance = is_variance
        self.num = contrast_num
        self.coord_space = coord_space
        if is_variance:
            self.var_coord_space = var_coord_space
        self.type = NIDM_CONTRAST_STANDARD_ERROR_MAP
        self.prov_type = PROV['Entity']
        self.format = format
        self.sha = sha
        self.filename = filename
        if label is None:
            self.label = "Contrast Map: " + self.name
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_ContrastStandardErrorMap: <http://purl.org/nidash/nidm#NIDM_0000013>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ContrastStandardErrorMap: ;
            rdfs:label ?label ;
            prov:atLocation ?filepath ;
            nfo:fileName ?filename ;
            dct:format ?format ;
            crypto:sha512 ?sha ;
        }
        """
        return query        

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """
        std_filename = "ContrastStandardError" + self.num + ".nii.gz"
        if self.is_variance:
            # Copy contrast variance map in export directory
            path, var_cope_filename = os.path.split(self.file)
            contrast_var = ContrastVariance(
                self.var_coord_space, self.file, var_cope_filename, format=self.format, sha=self.sha, filename=self.filename)
            self.contrast_var = contrast_var

            # Create standard error map from contrast variance map
            var_cope_img = nib.load(self.file)
            contrast_variance = var_cope_img.get_data()

            standard_error_img = nib.Nifti1Image(np.sqrt(contrast_variance),
                                                 var_cope_img.get_qform())

            stderr_file = os.path.join(export_dir, std_filename)
            nib.save(standard_error_img, stderr_file)
            self.file = NIDMFile(
                self.id, stderr_file, std_filename)

        else:
            self.file = NIDMFile(self.id, self.file, self.filename, format=self.format, sha=self.sha)

        self.add_attributes((
            (PROV['type'], self.type),
            (DCT['format'], "image/nifti"),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], "Contrast Standard Error Map")))


class ContrastVariance(NIDMObject):
    def __init__(self, coord_space, var_file, filename, format=None, 
                oid=None):
        super(ContrastVariance, self).__init__(oid=oid)
        self.id = NIIRI[str(uuid.uuid4())]
        self.coord_space = coord_space
        self.type = NIDM_CONTRAST_VARIANCE_MAP
        self.file = NIDMFile(self.id, var_file, format=format)
        self.filename = filename
        self.prov_type = PROV['Entity']

    def export(self, nidm_version, export_dir):
        self.add_attributes([(PROV['type'], NIDM_CONTRAST_VARIANCE_MAP)])


class StatisticMap(NIDMObject):

    """
    Object representing a StatisticMap entity.
    """

    def __init__(self, location, stat_type, contrast_name, dof, coord_space,
                 contrast_num=None, label=None, oid=None,
                 format="image/nifti", effdof=None, filename=None, sha=None,
                 contrast_estimation=None):
        super(StatisticMap, self).__init__(oid=oid)
        self.num = contrast_num
        self.contrast_name = contrast_name
        self.stat_type = stat_type
        if self.stat_type.lower() == "t":
            self.stat = STATO_TSTATISTIC
        elif self.stat_type.lower() == "z":
            self.stat = STATO_ZSTATISTIC
        elif self.stat_type.lower() == "f":
            self.stat = STATO_FSTATISTIC
        elif self.stat_type.startswith('http'):
            self.stat = Identifier(self.stat_type)
        else:
            raise Exception('Unrecognised statistic: ' + str(self.stat_type))

        # FIXME use new 'preferred mathematical notation from stato'
        if self.num is not None:
            filename = self.stat_type.upper() + \
                'Statistic' + self.num + '.nii.gz'
        self.file = NIDMFile(self.id, location, filename, sha=sha)
        self.coord_space = coord_space

        self.dof = dof
        self.type = NIDM_STATISTIC_MAP
        self.prov_type = PROV['Entity']
        if label is not None:
            self.label = label
        else:
            self.label = "Statistic Map: " + self.contrast_name
            # Include statistic type in the label
            self.label = self.stat_type + '-' + self.label

        self.format = format
        if effdof is None:
            # FIXME: this should not be 1 for F-test
            effdof = 1.0

        self.effdof = effdof

        # Only used when reading (so far)
        self.contrast_estimation = contrast_estimation

    def __str__(self):
        return '%s\t%s' % (self.label, self.file)

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix nidm_StatisticMap: <http://purl.org/nidash/nidm#NIDM_0000076>

        prefix nidm_statisticType: <http://purl.org/nidash/nidm#NIDM_0000123>
        prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>
        prefix nidm_effectDegreesOfFreedom: <http://purl.org/nidash/nidm#NIDM_0000091>
        prefix nidm_errorDegreesOfFreedom: <http://purl.org/nidash/nidm#NIDM_0000093>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_StatisticMap: ;
            rdfs:label ?label ;
            prov:atLocation ?location ;
            dct:format ?format ;
            nfo:fileName ?filename ;
            nidm_contrastName: ?contrast_name ;
            crypto:sha512 ?sha ;
            nidm_statisticType: ?stat_type ;
            nidm_effectDegreesOfFreedom: ?effdof ;
            nidm_errorDegreesOfFreedom: ?dof .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """
        attributes = [(PROV['type'], NIDM_STATISTIC_MAP),
                      (DCT['format'], self.format),
                      (PROV['label'], self.label),
                      (NIDM_STATISTIC_TYPE, self.stat),
                      (NIDM_CONTRAST_NAME, self.contrast_name),
                      (NIDM_IN_COORDINATE_SPACE, self.coord_space.id)]

        if not self.stat_type == 'Z':
            attributes.insert(0, (NIDM_ERROR_DEGREES_OF_FREEDOM, self.dof))
            attributes.insert(0, (NIDM_EFFECT_DEGREES_OF_FREEDOM, self.effdof))
        else:
            # For Z-Statistic error dof is infinity and effect dof is 1
            attributes.insert(0, (NIDM_ERROR_DEGREES_OF_FREEDOM, float("inf")))
            attributes.insert(0, (NIDM_EFFECT_DEGREES_OF_FREEDOM, self.effdof))

        # Create "Statistic Map" entity
        # FIXME: Deal with other than t-contrast maps: dof + statisticType
        self.add_attributes(attributes)


class ContrastEstimation(NIDMObject):

    """
    Object representing a ContrastEstimation entity.
    """

    def __init__(self, contrast_num, contrast_name=None, label=None,
                 param_estimate_id=None, oid=None):
        super(ContrastEstimation, self).__init__(oid=oid)
        self.num = contrast_num
        self.type = NIDM_CONTRAST_ESTIMATION
        self.prov_type = PROV['Activity']
        if label is not None:
            self.label = label
        else:
            self.label = "Contrast estimation: " + contrast_name
        # Only used when reading (so far)
        self.param_estimate_id = param_estimate_id

    @classmethod
    def get_query(klass, oid=None):
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
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['label'], self.label)))
