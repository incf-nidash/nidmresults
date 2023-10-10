"""
Generic objects supporting the classes defined in NIDM-Results.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""
from prov.model import Identifier
from prov.identifier import QualifiedName
import numpy as np
import os
from nidmresults.objects.constants import *
import nibabel as nib
import shutil
import hashlib
import uuid
import json
from prov.model import Literal
from prov.constants import XSD_STRING
import warnings
import zipfile


class NIDMObject(object):

    """
    Generic class, parent of all objects describing a NIDM entity, activity
    or agent
    """

    def __init__(self, oid=None):
        if oid is None:
            self.id = NIIRI[str(uuid.uuid4())]
        else:
            if not type(oid) is QualifiedName:
                oid = NIIRI.qname(Identifier(oid))
            self.id = oid

    # the next three methods build a .file property
    # with a dedicated function call on "set" that can
    # be used to implement external ID mappings
    # e.g. DataLad can replace '_map_fileid()' to provides
    # its internal file IDs based on the filename associated
    # with a NIDMFile object
    @property
    def file(self):
        return getattr(self, '_file', None)

    @file.setter
    def file(self, fileobj):
        if isinstance(fileobj, NIDMFile):
            self._map_fileid(fileobj)
        self._file = fileobj

    def _map_fileid(self, fileobj):
        pass

    def __str__(self):
        value = ""
        if hasattr(self, 'value'):
            value = ": " + self.value
        location = ""
        if hasattr(self, 'file'):
            if hasattr(self.file, 'path'):
                location = " - " + self.file.path
        return '"' + self.label + '"' + value + location

    def __repr__(self):
        return '<"' + self.label + '" ' + \
               str(self.id).replace("niiri:", "").replace(NIIRI._uri, "")[0:8]\
               + '>'

    def add_attributes(self, attributes):
        if hasattr(self, 'attributes'):
            if isinstance(attributes, tuple):
                attributes = list(attributes)
            if isinstance(self.attributes, tuple):
                self.attributes = list(self.attributes)

            if isinstance(attributes, dict):
                attributes = [[k, v] for k, v in attributes.items()]
            if isinstance(self.attributes, dict):
                self.attributes = [[k, v] for k, v in self.attributes.items()]

            self.attributes = attributes + self.attributes
        else:
            self.attributes = attributes

    @classmethod
    def load(klass, loaded_from, *args, **kwargs):
        if type(loaded_from) is dict:
            obj = klass.load_from_json(loaded_from, *args, **kwargs)
        return obj


class NIDMResultsBundle(NIDMObject):
    """
    Object representing a NIDM-Results bundle entity.
    """

    def __init__(self, nidm_version=None, label=None, oid=None):
        super(NIDMResultsBundle, self).__init__(oid=oid)
        self.type = NIDM_RESULTS
        self.nidm_version = nidm_version
        if label is None:
            self.label = "NIDM-Results"
        else:
            self.label = label
        self.prov_type = PROV['Bundle']

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix dctype: <http://purl.org/dc/dcmitype/>

SELECT * WHERE
{
    """ + oid_var + """ a nidm_NIDMResults: ;
    rdfs:label ?label ;
    nidm_version: ?nidm_version .
}
        """
        return query

    @classmethod
    def load_from_json(klass, json_dict):
        nidm_version = json_dict["NIDMResults_version"]
        bd = NIDMResultsBundle(nidm_version)

        return bd

    def export(self, nidm_version, export_dir):
        """
        Create prov entity.
        """
        self.add_attributes([
            (PROV['type'], self.type),
            (PROV['type'], PROV['Bundle']),
            # Explicitely add bundle type
            (PROV['label'], self.label),
            (NIDM_VERSION, self.nidm_version),
        ])


class CoordinateSpace(NIDMObject):
    """
    Object representing a CoordinateSpace entity.
    """

    def __init__(self, coordinate_system, nifti_file=None, vox_to_world=None,
                 vox_size=None, dimensions=None, numdim=None, units=None,
                 oid=None, label="Coordinate space"):
        super(CoordinateSpace, self).__init__(oid)
        
        if not isinstance(coordinate_system, QualifiedName):
            coordinate_system = NIDM.qname(coordinate_system)

        self.coordinate_system = coordinate_system
        self.type = NIDM_COORDINATE_SPACE
        self.prov_type = PROV['Entity']
        self.label = label

        if (vox_to_world is None) and (vox_size is None) and\
                (dimensions is None) and (numdim is None) and\
                (units is None) and \
                (nifti_file is not None):
            thresImg = nib.load(nifti_file)
            thresImgHdr = thresImg.get_header()

            numdim = len(thresImg.shape)
            dimensions = np.asarray(thresImg.shape)
            # FIXME: is vox_to_world the qform?
            vox_to_world = thresImg.get_qform()
            vox_size = thresImgHdr['pixdim'][1:(numdim + 1)]
            # FIXME: this gives mm, sec => what is wrong: FSL file, nibabel,
            # other?
            # units = str(thresImgHdr.get_xyzt_units()).strip('()')
            units = ["mm", "mm", "mm"]

        self.number_of_dimensions = numdim
        if not type(vox_to_world) is np.ndarray:
            # This is useful if info was read from a NIDM pack
            vox_to_world = np.array(json.loads(vox_to_world))
            dimensions = np.array(json.loads(dimensions))
            units = json.loads(units)
            vox_size = np.array(json.loads(vox_size))

        self.voxel_to_world = vox_to_world
        self.voxel_size = vox_size
        self.dimensions = dimensions
        self.units = units

    def is_mni(self):
        mni_coords = [
            NIDM_MNI_COORDINATE_SYSTEM.uri,
            NIDM_ICBM_MNI152_LINEAR_COORDINATE_SYSTEM.uri,
            NIDM_ICBM_MNI152_NON_LINEAR2009A_ASYMMETRIC_COORDINATE_SYSTEM.uri,
            NIDM_ICBM_MNI152_NON_LINEAR2009A_SYMMETRIC_COORDINATE_SYSTEM.uri,
            NIDM_ICBM_MNI152_NON_LINEAR2009B_ASYMMETRIC_COORDINATE_SYSTEM.uri,
            NIDM_ICBM_MNI152_NON_LINEAR2009B_SYMMETRIC_COORDINATE_SYSTEM.uri,
            NIDM_ICBM_MNI152_NON_LINEAR2009C_ASYMMETRIC_COORDINATE_SYSTEM.uri,
            NIDM_ICBM_MNI152_NON_LINEAR2009C_SYMMETRIC_COORDINATE_SYSTEM.uri,
            NIDM_ICBM_MNI152_NON_LINEAR6TH_GENERATION_COORDINATE_SYSTEM.uri,
            NIDM_ICBM452_AIR_COORDINATE_SYSTEM.uri,
            NIDM_ICBM452_WARP5_COORDINATE_SYSTEM.uri,
            NIDM_IXI549_COORDINATE_SYSTEM.uri,
            NIDM_MNI305_COORDINATE_SYSTEM.uri]

        if str(self.coordinate_system) in mni_coords:
            return True
        else:
            return False

    def is_talairach(self):
        if str(self.coordinate_system) in \
                [NIDM_TALAIRACH_COORDINATE_SYSTEM.uri]:
            return True
        else:
            return False

    @classmethod
    def load_from_json(klass, json_dict, nifti_file):
        COORD_SYS = {
            'MNICoordinateSystem': NIDM_MNI_COORDINATE_SYSTEM,
            'IcbmMni152LinearCoordinateSystem': NIDM_ICBM_MNI152_LINEAR_COORDINATE_SYSTEM,
            'IcbmMni152NonLinear2009aAsymmetricCoordinateSystem': NIDM_ICBM_MNI152_NON_LINEAR2009A_ASYMMETRIC_COORDINATE_SYSTEM,
            'IcbmMni152NonLinear2009aSymmetricCoordinateSystem': NIDM_ICBM_MNI152_NON_LINEAR2009A_SYMMETRIC_COORDINATE_SYSTEM,
            'IcbmMni152NonLinear2009bAsymmetricCoordinateSystem': NIDM_ICBM_MNI152_NON_LINEAR2009B_ASYMMETRIC_COORDINATE_SYSTEM,
            'IcbmMni152NonLinear2009bSymmetricCoordinateSystem': NIDM_ICBM_MNI152_NON_LINEAR2009B_SYMMETRIC_COORDINATE_SYSTEM,
            'IcbmMni152NonLinear2009cAsymmetricCoordinateSystem': NIDM_ICBM_MNI152_NON_LINEAR2009C_ASYMMETRIC_COORDINATE_SYSTEM,
            'IcbmMni152NonLinear2009cSymmetricCoordinateSystem': NIDM_ICBM_MNI152_NON_LINEAR2009C_SYMMETRIC_COORDINATE_SYSTEM,
            'IcbmMni152NonLinear6thGenerationCoordinateSystem': NIDM_ICBM_MNI152_NON_LINEAR6TH_GENERATION_COORDINATE_SYSTEM,
            'Icbm452AirCoordinateSystem': NIDM_ICBM452_AIR_COORDINATE_SYSTEM,
            'Icbm452Warp5CoordinateSystem': NIDM_ICBM452_WARP5_COORDINATE_SYSTEM,
            'Ixi549CoordinateSystem': NIDM_IXI549_COORDINATE_SYSTEM,
            'Mni305CoordinateSystem': NIDM_MNI305_COORDINATE_SYSTEM,
            'TalairachCoordinateSystem': NIDM_TALAIRACH_COORDINATE_SYSTEM,
            'SubjectCoordinateSystem': NIDM_SUBJECT_COORDINATE_SYSTEM,
            'CustomCoordinateSystem': NIDM_CUSTOM_COORDINATE_SYSTEM,
        }      
        coordsys = COORD_SYS[
            json_dict['CoordinateSpace_inWorldCoordinateSystem']]
        coord_space = CoordinateSpace(coordsys, nifti_file)
        
        return coord_space

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_CoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000016>
prefix nidm_voxelToWorldMapping: <http://purl.org/nidash/nidm#NIDM_0000132>
prefix nidm_voxelUnits: <http://purl.org/nidash/nidm#NIDM_0000133>
prefix nidm_voxelSize: <http://purl.org/nidash/nidm#NIDM_0000131>
prefix nidm_inWorldCoordinateSystem: <http://purl.org/nidash/nidm#NIDM_0000105>
prefix nidm_MNICoordinateSystem: <http://purl.org/nidash/nidm#NIDM_0000051>
prefix nidm_numberOfDimensions: <http://purl.org/nidash/nidm#NIDM_0000112>
prefix nidm_dimensionsInVoxels: <http://purl.org/nidash/nidm#NIDM_0000090>


SELECT ?oid ?label ?vox_to_world ?units ?vox_size ?coordinate_system ?numdim
?dimensions
        WHERE
        {
    """ + oid_var + """ a nidm_CoordinateSpace: ;
    rdfs:label ?label ;
    nidm_voxelToWorldMapping: ?vox_to_world ;
    nidm_voxelUnits: ?units ;
    nidm_voxelSize: ?vox_size ;
    nidm_inWorldCoordinateSystem: ?coordinate_system ;
    nidm_numberOfDimensions: ?numdim ;
    nidm_dimensionsInVoxels: ?dimensions .
    }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        self.add_attributes({
            PROV['type']: self.type,
            NIDM_DIMENSIONS_IN_VOXELS: json.dumps(self.dimensions.tolist()),
            NIDM_NUMBER_OF_DIMENSIONS: self.number_of_dimensions,
            NIDM_VOXEL_TO_WORLD_MAPPING:
            json.dumps(self.voxel_to_world.tolist()),
            NIDM_IN_WORLD_COORDINATE_SYSTEM: self.coordinate_system,
            NIDM_VOXEL_UNITS: json.dumps(self.units),
            NIDM_VOXEL_SIZE: json.dumps(self.voxel_size.tolist()),
            PROV['label']: self.label})


class NIDMFile(NIDMObject):
    """
    Object representing a File (to be used as attribute of another class)
    """
    def __init__(self, rdf_id, location, filename=None,
                 sha=None, fmt=None, temporary=False):
        super(NIDMFile, self).__init__()
        self.prov_type = PROV['Entity']
        self.path = location
        if filename is None:
            # Keep same file name
            path, self.filename = os.path.split(self.path)
        else:
            self.filename = filename

        # NIDMFile is not a NIDM class defined in the owl file
        self.type = None
        self.id = rdf_id
        self.label = "'NIDM file'"  # used if display is called

        self.sha = sha
        self.fmt = fmt
        self.temporary = temporary

    def is_nifti(self):
        if self.path is not None:
            name = self.path
        else:
            name = self.filename

        return name.endswith(".nii") or \
            name.endswith(".nii.gz") or \
            name.endswith(".img") or \
            name.endswith(".hrd")

    def get_sha_sum(self, nifti_file):
        nifti_img = nib.load(nifti_file)
        data = nifti_img.get_data()
        # Fix needed as in https://github.com/pymc-devs/pymc/issues/327
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        return hashlib.sha512(data).hexdigest()

    def export(self, nidm_version, export_dir, prepend_path):
        """
        Copy file over of export_dir and create corresponding triples
        """
        if self.path is not None:
            if export_dir is not None:
                # Copy file only if export_dir is not None
                new_file = os.path.join(export_dir, self.filename)
                if not self.path == new_file:
                    if prepend_path.endswith('.zip'):
                        with zipfile.ZipFile(prepend_path) as z:
                            extracted = z.extract(str(self.path), export_dir)
                            shutil.move(extracted, new_file)
                    else:
                        if prepend_path:
                            file_copied = os.path.join(prepend_path, self.path)
                        else:
                            file_copied = self.path
                        shutil.copy(file_copied, new_file)

                    if self.temporary:
                        os.remove(self.path)

            else:
                new_file = self.path

        if nidm_version['num'] in ["1.0.0", "1.1.0"]:
            loc = Identifier("file://./" + self.filename)
        else:
            loc = Identifier(self.filename)

        self.add_attributes([(NFO['fileName'], self.filename)])

        if export_dir:
            self.add_attributes([(PROV['atLocation'], loc)])

        if nidm_version['num'] in ("1.0.0", "1.1.0"):
            path, org_filename = os.path.split(self.path)
            if (org_filename is not self.filename) \
                    and (not self.temporary):
                self.add_attributes([(NFO['fileName'], org_filename)])

        if self.is_nifti():
            if self.sha is None:
                self.sha = self.get_sha_sum(new_file)
            if self.fmt is None:
                self.fmt = "image/nifti"

            self.add_attributes([
                (CRYPTO['sha512'], self.sha),
                (DCT['format'], self.fmt)
            ])


class Image(NIDMObject):

    """
    Object representing an Image entity.
    """

    def __init__(self, image_file, filename, fmt='png', oid=None):
        super(Image, self).__init__(oid=oid)
        self.type = DCTYPE['Image']
        self.prov_type = PROV['Entity']
        self.file = NIDMFile(self.id, image_file, filename)
        self.label = ""  # Enable printing

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
        prefix dctype: <http://purl.org/dc/dcmitype/>


        SELECT * WHERE
                {
            """ + oid_var + """ a dctype:Image ;
            prov:atLocation ?image_file ;
            nfo:fileName ?filename ;
            dct:format ?fmt .
            }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entity.
        """
        if self.file is not None:
            self.add_attributes([
                (PROV['type'], self.type),
                (DCT['format'], "image/png"),
            ])


class NeuroimagingSoftware(NIDMObject):
    """
    Class representing a NeuroimagingSoftware Agent.
    """

    def __init__(self, software_type, version, label=None, feat_version=None,
                 oid=None):
        super(NeuroimagingSoftware, self).__init__(oid=oid)
        self.version = version

        if isinstance(software_type, QualifiedName):
            self.type = software_type
        else:
            if software_type.startswith('http'):
                self.type = Identifier(software_type)
            elif software_type.lower() == "fsl":
                self.type = SCR_FSL
            elif software_type.lower() == "spm":
                self.type = SCR_SPM
            else:
                warnings.warn('Unrecognised software: ' + str(software_type))
                self.name = str(software_type)
                self.type = None

        # FIXME: get label from owl!
        if self.type == SCR_FSL:
            self.name = "FSL"
        elif self.type == SCR_SPM:
            self.name = "SPM"
        else:
            warnings.warn('Unrecognised software: ' + str(software_type))
            self.name = str(software_type)

        if not label:
            self.label = self.name
        else:
            self.label = label
        self.prov_type = PROV['Agent']
        self.feat_version = feat_version

    @classmethod
    def load_from_json(klass, json_dict):
        soft_type = json_dict['NeuroimagingAnalysisSoftware_type']
        version = json_dict['NeuroimagingAnalysisSoftware_type']
        label = json_dict.get('NeuroimagingAnalysisSoftware_label', None)
        soft = NeuroimagingSoftware(soft_type, version, label)
        return soft

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_softwareVersion: <http://purl.org/nidash/nidm#NIDM_0000122>
prefix fsl_featVersion: <http://purl.org/nidash/fsl#FSL_0000005>
prefix nidm_ModelParametersEstimation: <http://purl.org/nidash/nidm#NIDM_00000\
56>

SELECT DISTINCT * WHERE
        {
    """ + oid_var + """ a prov:SoftwareAgent ;
        nidm_softwareVersion: ?version .

    [] a nidm_ModelParametersEstimation: ;
        prov:wasAssociatedWith """ + oid_var + """ .

    OPTIONAL {""" + oid_var + """ a ?software_type .} .
    OPTIONAL {""" + oid_var + """ fsl_featVersion: ?feat_version .} .

    FILTER ( ?software_type NOT IN (prov:SoftwareAgent, prov:Agent) )
    }
        """
        return query

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        if nidm_version['major'] < 1 or \
                (nidm_version['major'] == 1 and nidm_version['minor'] < 3):
            self.type = NLX_OLD_FSL

        atts = (
            (PROV['type'], self.type),
            (PROV['type'], PROV['SoftwareAgent']),
            (PROV['label'], Literal(self.label, datatype=XSD_STRING)),
            (NIDM_SOFTWARE_VERSION, self.version)
            )

        if self.feat_version:
            atts = atts + ((FSL_FEAT_VERSION, self.feat_version),)

        self.add_attributes(atts)


class ExporterSoftware(NIDMObject):
    """
    Class representing a Software Agent.
    """

    def __init__(self, software_type, version, oid=None, label=None):
        super(ExporterSoftware, self).__init__(oid=oid)
        self.type = software_type
        self.prov_type = PROV['Agent']
        self.version = version

        if label is None:
            if software_type == NIDM_FSL:
                self.label = "nidmfsl"
            else:
                self.label = str(software_type)
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_softwareVersion: <http://purl.org/nidash/nidm#NIDM_0000122>
prefix nidm_NIDMResultsExport: <http://purl.org/nidash/nidm#NIDM_0000166>

SELECT DISTINCT * WHERE
        {
    """ + oid_var + """ a prov:SoftwareAgent ;
        rdfs:label  ?label ;
        rdf:type ?software_type ;
        nidm_softwareVersion: ?version .

    FILTER ( ?software_type NOT IN (prov:SoftwareAgent, prov:Agent) )
    }
        """
        return query

    @classmethod
    def load_from_json(klass, json_dict):
        software_type = json_dict["NIDMResultsExporter_type"]
        version = json_dict["NIDMResultsExporter_softwareVersion"]
        exp = ExporterSoftware(software_type, version)

        return exp

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['type'], PROV['SoftwareAgent']),
            (PROV['label'], self.label),
            (NIDM_SOFTWARE_VERSION, self.version))
        )


class NIDMResultsExport(NIDMObject):
    """
    Class representing a NIDM-Results Export activity.
    """
    def __init__(self, oid=None, label=None):
        super(NIDMResultsExport, self).__init__(oid=oid)
        self.type = NIDM_NIDM_RESULTS_EXPORT
        self.prov_type = PROV['Activity']
        if label is None:
            self.label = "NIDM-Results export"
        else:
            self.label = label

    @classmethod
    def get_query(klass, oid=None):
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_NIDMResultsExport: <http://purl.org/nidash/nidm#NIDM_0000166>

SELECT DISTINCT * WHERE
    {
    """ + oid_var + """ a nidm_NIDMResultsExport: ;
        rdfs:label ?label .
    }
        """
        return query

    @classmethod
    def load_from_json(klass, json_dict):
        software_type = json_dict["NIDMResultsExporter_type"]
        version = json_dict["NIDMResultsExporter_softwareVersion"]
        exp = ExporterSoftware(software_type, version)

        return exp

    def export(self, nidm_version, export_dir):
        """
        Create prov entities and activities.
        """
        self.add_attributes([
            (PROV['label'], self.label),
            (PROV['type'], self.type)])
