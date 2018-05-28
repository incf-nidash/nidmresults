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
from prov.identifier import QualifiedName


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
        if not type(contrast_weights) is list:
            contrast_weights = json.loads(contrast_weights)
        self.contrast_weights = contrast_weights
        self.contrast_num = contrast_num
        self.stat_type = stat_type
        self.type = STATO_CONTRAST_WEIGHT_MATRIX
        self.prov_type = PROV['Entity']
        if label is None:
            self.label = "Contrast Weights: " + self.contrast_name
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
        self.stat = None
        if isinstance(self.stat_type, QualifiedName):
            stat = self.stat_type
        elif self.stat_type is not None:
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
            (PROV['value'], json.dumps(self.contrast_weights))))


class ContrastMap(NIDMObject):

    """
    Object representing a ContrastMap entity.
    """

    def __init__(self, contrast_file, contrast_num, contrast_name,
                 coord_space, sha=None, fmt=None,
                 label=None, filename=None, oid=None, derfrom_id=None,
                 derfrom_filename=None, derfrom_fmt=None,
                 derfrom_sha=None, isderfrommap=False):
        super(ContrastMap, self).__init__(oid=oid)
        self.num = contrast_num
        self.name = contrast_name
        if filename is None:
            filename = 'Contrast' + self.num + '.nii.gz'
        self.file = NIDMFile(self.id, contrast_file, filename, sha=sha,
                             fmt=fmt)
        self.coord_space = coord_space
        self.type = NIDM_CONTRAST_MAP
        self.prov_type = PROV['Entity']
        if label is None:
            if self.name:
                self.label = "Contrast Map: " + self.name
            else:
                self.label = None
        else:
            self.label = label

        if derfrom_id is not None:
            self.derfrom = ContrastMap(
                contrast_file=None, contrast_num=None,
                contrast_name=None, oid=derfrom_id, coord_space=coord_space,
                filename=derfrom_filename, sha=derfrom_sha,
                fmt=derfrom_fmt, isderfrommap=True)
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
        prefix nidm_ContrastMap: <http://purl.org/nidash/nidm#NIDM_0000002>

        SELECT DISTINCT * WHERE {
            """ + oid_var + """ a nidm_ContrastMap: ;
            rdfs:label ?label ;
            prov:atLocation ?contrast_file ;
            dct:format ?fmt ;
            nfo:fileName ?filename ;
            nidm_contrastName: ?contrast_name ;
            crypto:sha512 ?sha .

            OPTIONAL {""" + oid_var + """ prov:wasDerivedFrom ?derfrom_id .

            ?derfrom_id a nidm_ContrastMap: ;
                nfo:fileName ?derfrom_filename ;
                dct:format ?derfrom_fmt ;
                crypto:sha512 ?derfrom_sha .
             } .
        }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """
        # Contrast Map entity
        atts = (
            (PROV['type'], NIDM_CONTRAST_MAP),
            (NIDM_CONTRAST_NAME, self.name))

        if not self.isderfrommap:
            atts = atts + (
                (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),)

        if self.label is not None:
            atts = atts + (
                (PROV['label'], self.label),)

        if self.name is not None:
            atts = atts + (
                (NIDM_CONTRAST_NAME, self.name),)

        # Parameter estimate entity
        self.add_attributes(atts)


class ContrastExplainedMeanSquareMap(NIDMObject):
    """
    Object representing a ContrastExplainedMeanSquareMap entity.
    """
    def __init__(self, stat_file, sigma_sq_file, contrast_num,
                 coord_space, expl_mean_sq_file=None,
                 sha=None, fmt=None, filename=None, oid=None,
                 label="Contrast Explained Mean Square Map"):
        super(ContrastExplainedMeanSquareMap, self).__init__(oid=oid)
        self.stat_file = stat_file
        self.sigma_sq_file = sigma_sq_file
        self.num = contrast_num
        self.coord_space = coord_space
        self.type = NIDM_CONTRAST_EXPLAINED_MEAN_SQUARE_MAP
        self.prov_type = PROV['Entity']
        self.expl_mean_sq_file = expl_mean_sq_file
        self.sha = sha
        self.filename = filename
        self.fmt = fmt
        self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_ContrastExplainedMeanSquareMap: <http://purl.org/nidash/nidm#NIDM_\
0000163>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_ContrastExplainedMeanSquareMap: ;
    rdfs:label ?label ;
    prov:atLocation ?expl_mean_sq_file ;
    dct:format ?fmt ;
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

            expl_mean_sq = nib.Nifti1Image(
                fstat*sigma_sq, fstat_img.get_qform())

            self.filename = ("ContrastExplainedMeanSquareMap" +
                             self.num + ".nii.gz")
            self.expl_mean_sq_file = os.path.join(
                export_dir, self.filename)
            nib.save(expl_mean_sq, self.expl_mean_sq_file)

        self.file = NIDMFile(self.id, self.expl_mean_sq_file,
                             filename=self.filename,
                             sha=self.sha, fmt=self.fmt)

        # Contrast Explained Mean Square Map entity
        path, filename = os.path.split(self.expl_mean_sq_file)
        self.add_attributes((
            (PROV['type'], self.type),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], self.label)))


class ContrastStdErrMap(NIDMObject):

    """
    Object representing a ContrastStdErrMap entity.
    """
# stderr_or_expl_mean_sq_map = self.get_object(
# ContrastStdErrMap, args['constdm_id'],
#  coord_space=contraststd_map_coordspace,
# contrast_num=contrast_num, is_variance=False, var_coord_space=None)

    def __init__(self, contrast_num, filepath, is_variance, coord_space,
                 var_coord_space, label=None, fmt=None,
                 sha=None, filename=None, oid=None, derfrom_id=None,
                 derfrom_filename=None, derfrom_sha=None, derfrom_fmt=None,
                 export_dir=None):
        super(ContrastStdErrMap, self).__init__(oid=oid)
        self.file = filepath
        self.is_variance = is_variance
        self.num = contrast_num
        self.coord_space = coord_space
        if is_variance or derfrom_id:
            self.var_coord_space = var_coord_space
        self.type = NIDM_CONTRAST_STANDARD_ERROR_MAP
        self.prov_type = PROV['Entity']
        self.fmt = fmt
        self.sha = sha
        self.filename = filename

        std_filename = "ContrastStandardError" + self.num + ".nii.gz"
        if self.is_variance:
            # Copy contrast variance map in export directory
            path, var_cope_filename = os.path.split(self.file)
            contrast_var = ContrastVariance(
                coord_space=self.var_coord_space, var_file=self.file, 
                filename=var_cope_filename, fmt=self.fmt, sha=self.sha,
                oid=derfrom_id)
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
            self.file = NIDMFile(self.id, self.file, self.filename,
                                 fmt=self.fmt, sha=self.sha)

            if derfrom_id is not None:
                # TODO: assuming same coordinate space for derived from
                self.contrast_var = ContrastVariance(
                    coord_space=self.coord_space, var_file=None,
                    filename=derfrom_filename, fmt=derfrom_fmt,
                    sha=derfrom_sha, oid=derfrom_id)
            else:
                self.contrast_var = None

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_ContrastStandardErrorMap: <http://purl.org/nidash/nidm#NIDM_000001\
3>
prefix nidm_ContrastVarianceMap: <http://purl.org/nidash/nidm#NIDM_0000135>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_ContrastStandardErrorMap: ;
    rdfs:label ?label ;
    prov:atLocation ?filepath ;
    nfo:fileName ?filename ;
    dct:format ?fmt ;
    crypto:sha512 ?sha .

    OPTIONAL {""" + oid_var + """ prov:wasDerivedFrom ?derfrom_id .

    ?derfrom_id a nidm_ContrastVarianceMap: ;
        nfo:fileName ?derfrom_filename ;
        dct:format ?derfrom_fmt ;
        crypto:sha512 ?derfrom_sha .
     } .
}
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """

        self.add_attributes((
            (PROV['type'], self.type),
            (DCT['format'], "image/nifti"),
            (NIDM_IN_COORDINATE_SPACE, self.coord_space.id),
            (PROV['label'], "Contrast Standard Error Map")))


class ContrastVariance(NIDMObject):
    def __init__(self, coord_space, var_file, filename, fmt=None,
                 sha=None, oid=None):
        super(ContrastVariance, self).__init__(oid=oid)
        self.coord_space = coord_space
        self.type = NIDM_CONTRAST_VARIANCE_MAP
        self.filename = filename
        self.fmt = fmt
        self.sha = sha
        self.file = NIDMFile(self.id, var_file, filename=self.filename,
                             fmt=self.fmt, sha=self.sha)
        self.prov_type = PROV['Entity']

    def export(self, nidm_version, export_dir):
        self.add_attributes([(PROV['type'], NIDM_CONTRAST_VARIANCE_MAP)])


class StatisticMap(NIDMObject):

    """
    Object representing a StatisticMap entity.
    """

    def __init__(self, location, stat_type, contrast_name, dof, coord_space,
                 contrast_num=None, label=None, oid=None,
                 fmt="image/nifti", effdof=None, filename=None, sha=None,
                 contrast_estimation=None, derfrom_id=None,
                 derfrom_filename=None, derfrom_fmt=None,
                 derfrom_sha=None, isderfrommap=False):
        super(StatisticMap, self).__init__(oid=oid)
        self.num = contrast_num
        self.contrast_name = contrast_name
        self.stat_type = stat_type

        self.stat = None
        if isinstance(self.stat_type, QualifiedName):
            self.stat = self.stat_type
        elif self.stat_type is not None:
            if self.stat_type.lower() == "t":
                self.stat = STATO_TSTATISTIC
            elif self.stat_type.lower() == "z":
                self.stat = STATO_ZSTATISTIC
            elif self.stat_type.lower() == "f":
                self.stat = STATO_FSTATISTIC
            elif self.stat_type.startswith('http'):
                self.stat = Identifier(self.stat_type)
            else:
                raise Exception(
                    'Unrecognised statistic: ' + str(self.stat_type))

        if derfrom_id is not None:
            self.derfrom = StatisticMap(
                None, None, None, None,
                coord_space=None, oid=derfrom_id,
                filename=derfrom_filename, sha=derfrom_sha,
                fmt=derfrom_fmt,
                isderfrommap=True)
        else:
            self.derfrom = None

        # FIXME use new 'preferred mathematical notation from stato'
        if self.num is not None:
            filename = self.stat_type.upper() + \
                'Statistic' + self.num + '.nii.gz'
        self.file = NIDMFile(self.id, location, filename, sha=sha)
        self.coord_space = coord_space

        self.dof = dof
        self.type = NIDM_STATISTIC_MAP
        self.prov_type = PROV['Entity']
        self.label = label
        if label is not None:
            self.label = label
        else:
            if self.contrast_name:
                self.label = "Statistic Map: " + self.contrast_name
                # Include statistic type in the label
                self.label = self.stat_type + '-' + self.label

        self.fmt = fmt

        # Effect degrees of freedom for T-test is always 1
        if (effdof is None) and (self.stat in
                                 [STATO_TSTATISTIC, STATO_ZSTATISTIC]):
            effdof = 1.0

        self.effdof = effdof

        # Only used when reading (so far)
        self.contrast_estimation = contrast_estimation

        self.isderfrommap = isderfrommap

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
    dct:format ?fmt ;
    nfo:fileName ?filename ;
    nidm_contrastName: ?contrast_name ;
    crypto:sha512 ?sha ;
    nidm_statisticType: ?stat_type ;
    nidm_effectDegreesOfFreedom: ?effdof ;
    nidm_errorDegreesOfFreedom: ?dof .

    OPTIONAL {""" + oid_var + """ prov:wasDerivedFrom ?derfrom_id .

    ?derfrom_id a nidm_StatisticMap: ;
        nfo:fileName ?derfrom_filename ;
        dct:format ?derfrom_fmt ;
        crypto:sha512 ?derfrom_sha .
     } .
}
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov graph.
        """
        attributes = [(PROV['type'], NIDM_STATISTIC_MAP),
                      (DCT['format'], self.fmt)]

        if not self.isderfrommap:
            attributes.insert(0, (
                NIDM_IN_COORDINATE_SPACE,  self.coord_space.id))
            attributes.insert(0, (PROV['label'], self.label))

        if not self.stat_type == 'Z':
            attributes.insert(0, (NIDM_ERROR_DEGREES_OF_FREEDOM, self.dof))
            attributes.insert(0, (NIDM_EFFECT_DEGREES_OF_FREEDOM, self.effdof))
        else:
            # For Z-Statistic error dof is infinity and effect dof is 1
            attributes.insert(0, (NIDM_ERROR_DEGREES_OF_FREEDOM, float("inf")))
            attributes.insert(0, (NIDM_EFFECT_DEGREES_OF_FREEDOM, self.effdof))

        if self.stat is not None:
            attributes.insert(0, (NIDM_STATISTIC_TYPE, self.stat))

        if self.contrast_name is not None:
            attributes.insert(0, (NIDM_CONTRAST_NAME, self.contrast_name))

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
