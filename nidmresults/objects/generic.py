"""
Generic objects supporting the classes defined in NIDM-Results.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""
from prov.model import ProvBundle, Identifier
import prov
import numpy as np
import os
from constants import *
import nibabel as nib
import shutil
import hashlib
import uuid
import rdflib
import json
from rdflib.namespace import RDF, RDFS, XSD


class NIDMObject(object):

    """
    Generic class, parent of all objects describing a NIDM entity, activity
    or agent
    """

    def __init__(self, export_dir=None, oid=None):
        self.export_dir = export_dir
        self.p = ProvBundle()

        self.g = rdflib.Graph()
        if oid is None:
            self.id = NIIRI[str(uuid.uuid4())]
        else:
            self.id = oid

    def __str__(self):
        value = ""
        if hasattr(self, 'value'):
            value = ": " + self.value
        return '"' + self.label + '"' + value

    def __repr__(self):
        return '<"' + self.label + '" ' + \
               str(self.id).replace("niiri:", "").replace(NIIRI._uri, "")[0:8]\
               + '>'

    def _rdf_add_attributes(self, attributes):
        if self.type is not None:
            self._rdf_add(self.id, RDF.type, self.type)
        self._rdf_add(self.id, RDF.type, self.prov_type)

        # If attributes is a dictionnary, convert to list
        if isinstance(attributes, dict):
            attributes = [(k, v) for k, v in attributes.iteritems()]

        for p, o in attributes:
            self._rdf_add(self.id, p, o)

    def _rdf_add(self, s, p, o):
        if isinstance(o, prov.identifier.QualifiedName):
            o = rdflib.URIRef(o.uri)
        elif isinstance(o, prov.identifier.Identifier):
            o = rdflib.Literal(o, datatype=XSD.anyURI)
        else:
            if (isinstance(o, str) or isinstance(o, unicode)) and \
                    not str(p) == "prov:label":
                o = rdflib.Literal(o, datatype=XSD.string)
            elif isinstance(o, float):
                o = rdflib.Literal(o, datatype=XSD.float)
            elif isinstance(o, bool):
                o = rdflib.Literal(o, datatype=XSD.boolean)
            elif isinstance(o, int):
                o = rdflib.Literal(o, datatype=XSD.int)
            else:
                o = rdflib.Literal(str(o))

        if not isinstance(p, rdflib.URIRef):
            if str(p) == "prov:type":
                p = RDF.type
            elif str(p) == "prov:label":
                p = RDFS.label
            elif str(p) == "prov:location":
                p = rdflib.URIRef(p.uri.replace("location", "atLocation"))
            else:
                p = rdflib.URIRef(p.uri)

        self.g.add((rdflib.URIRef(s.uri), p, o))

    def add_object(self, nidm_object, nidm_version):
        nidm_object.export(nidm_version)

        # Prov graph (=> provn)
        self.p.update(nidm_object.p)
        # RDF graph (=> turtle)
        self.g = self.g + nidm_object.g

    def used(self, nidm_object):
        self._add_prov_relation(PROV['used'], nidm_object)

    def wasGeneratedBy(self, nidm_object):
        self._add_prov_relation(PROV['wasGeneratedBy'], nidm_object)

    def wasDerivedFrom(self, nidm_object):
        self._add_prov_relation(PROV['wasDerivedFrom'], nidm_object)

    def wasAttributedTo(self, nidm_object):
        self._add_prov_relation(PROV['wasAttributedTo'], nidm_object)

    def wasAssociatedWith(self, nidm_object):
        self._add_prov_relation(PROV['wasAssociatedWith'], nidm_object)

    def _add_prov_relation(self, relation, nidm_object):
        if isinstance(nidm_object, NIDMObject):
            object_id = nidm_object.id
        else:
            object_id = nidm_object

        if relation == PROV['used']:
            self.p.used(self.id, object_id)
        elif relation == PROV['wasGeneratedBy']:
            self.p.wasGeneratedBy(self.id, object_id)
        elif relation == PROV['wasDerivedFrom']:
            self.p.wasDerivedFrom(self.id, object_id)
        elif relation == PROV['wasAssociatedWith']:
            self.p.wasAssociatedWith(self.id, object_id)
        elif relation == PROV['wasAttributedTo']:
            self.p.wasAttributedTo(self.id, object_id)
        else:
            raise Exception('Unrecognised prov relation')

        self._rdf_add(self.id, relation, object_id)

    def add_attributes(self, attributes):
        if self.prov_type == PROV['Activity']:
            self.p.activity(self.id, other_attributes=attributes)
        elif self.prov_type == PROV['Entity']:
            self.p.entity(self.id, other_attributes=attributes)
        elif self.prov_type == PROV['Agent']:
            self.p.agent(self.id, other_attributes=attributes)

        self._rdf_add_attributes(attributes)


# class NIDMBundle(NIDMObject):
#     """
#     Object representing a NIDM Bundle entity.
#     """

#     def __init__(self, version):
#         self.id = NIIRI[str(uuid.uuid4())]
#         self.version = version
#         self.type = NIDM_RESULTS
#         self.prov_type = PROV['Bundle']

#     def export(self):
#         self.bundle = ProvBundle(identifier=bundle_id)

#         self.doc.entity(bundle_id,
#                         other_attributes=((PROV['type'], PROV['Bundle'],),
#                                           (PROV['label'], "NIDM-Results"),
#                                           (PROV['type'], NIDM_RESULTS),
#                                           (NIDM_VERSION, version))
#                         )

#         self.doc.wasGeneratedBy(bundle_id,
#                                 time=str(datetime.datetime.now().time()))

class CoordinateSpace(NIDMObject):
    """
    Object representing a CoordinateSpace entity.
    """

    def __init__(self, coordinate_system, nifti_file=None, vox_to_world=None,
                 vox_size=None, dimensions=None, numdim=None, units=None,
                 oid=None, label="Coordinate space"):
        super(CoordinateSpace, self).__init__(oid)
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
            dimensions = thresImg.shape
            # FIXME: is vox_to_world the qform?
            vox_to_world = thresImg.get_qform()
            vox_size = thresImgHdr['pixdim'][1:(numdim + 1)]
            # FIXME: this gives mm, sec => what is wrong: FSL file, nibabel,
            # other?
            # units = str(thresImgHdr.get_xyzt_units()).strip('()')
            units = ["mm", "mm", "mm"]

        self.number_of_dimensions = numdim
        self.voxel_to_world = vox_to_world
        self.voxel_size = vox_size
        self.dimensions = dimensions
        self.units = units

    def is_mni(self):
        if str(self.coordinate_system) in [
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
                NIDM_MNI305_COORDINATE_SYSTEM.uri]:
            return True
        else:
            return False

    def is_talairach(self):
        if str(self.coordinate_system) in [NIDM_TALAIRACH_COORDINATE_SYSTEM.uri]:
            return True
        else:
            return False

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes({
            PROV['type']: self.type,
            NIDM_DIMENSIONS_IN_VOXELS: json.dumps(self.dimensions),
            NIDM_NUMBER_OF_DIMENSIONS: self.number_of_dimensions,
            NIDM_VOXEL_TO_WORLD_MAPPING:
            json.dumps(self.voxel_to_world.tolist()),
            NIDM_IN_WORLD_COORDINATE_SYSTEM: self.coordinate_system,
            NIDM_VOXEL_UNITS: json.dumps(self.units),
            NIDM_VOXEL_SIZE: json.dumps(self.voxel_size.tolist()),
            PROV['label']: self.label})
        return self.p


class NIDMFile(NIDMObject):
    """
    Object representing a File (to be used as attribute of another class)
    """
    def __init__(self, rdf_id, location, new_filename=None, export_dir=None,
                 sha=None, format=None, temporary=False):
        super(NIDMFile, self).__init__(export_dir)
        self.prov_type = PROV['Entity']
        self.path = location
        if new_filename is None:
            # Keep same file name
            path, self.new_filename = os.path.split(self.path)
        else:
            self.new_filename = new_filename

        # NIDMFile is not a NIDM class defined in the owl file
        self.type = None
        self.id = rdf_id
        self.label = "'NIDM file'"  # used if display is called

        self.sha = sha
        self.format = format
        self.temporary = temporary

    def is_nifti(self):
        return self.path.endswith(".nii") or \
            self.path.endswith(".nii.gz") or \
            self.path.endswith(".img") or \
            self.path.endswith(".hrd")

    def get_sha_sum(self, nifti_file):
        nifti_img = nib.load(nifti_file)
        data = nifti_img.get_data()
        # Fix needed as in https://github.com/pymc-devs/pymc/issues/327
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        return hashlib.sha512(data).hexdigest()

    def export(self, nidm_version):
        """
        Copy file over of export_dir and create corresponding triples
        """
        if self.path is not None:
            path, org_filename = os.path.split(self.path)
            if self.export_dir is not None:
                # Copy file only if export_dir is not None
                new_file = os.path.join(self.export_dir, self.new_filename)
                if not self.path == new_file:
                    shutil.copy(self.path, new_file)
                    if self.temporary:
                        os.remove(self.path)
            else:
                new_file = self.path

            if nidm_version['num'] in ["1.0.0", "1.1.0"]:
                loc = Identifier("file://./" + self.new_filename)
            else:
                loc = Identifier(self.new_filename)

            self.add_attributes([(NFO['fileName'], self.new_filename)])

            if self.export_dir:
                self.add_attributes([(PROV['atLocation'], loc)])

            if nidm_version['num'] in ("1.0.0", "1.1.0"):
                if org_filename is not self.new_filename:
                    self.add_attributes([(NFO['fileName'], org_filename)])

            if self.is_nifti():
                if self.sha is None:
                    self.sha = self.get_sha_sum(new_file)
                if self.format is None:
                    self.format = "image/nifti"

                self.add_attributes([
                    (CRYPTO['sha512'], self.sha),
                    (DCT['format'], self.format)
                ])

            return self.p


class Image(NIDMObject):

    """
    Object representing an Image entity.
    """

    def __init__(self, export_dir, image_file):
        super(Image, self).__init__(export_dir)
        self.type = DCTYPE['Image']
        self.prov_type = PROV['Entity']
        self.id = NIIRI[str(uuid.uuid4())]
        filename = "DesignMatrix.png"
        self.file = NIDMFile(self.id, image_file, filename, export_dir)

    def export(self, nidm_version):
        """
        Create prov entity.
        """
        if self.file is not None:
            self.add_object(self.file, nidm_version)

            self.add_attributes([
                (PROV['type'], self.type),
                (DCT['format'], "image/png"),
            ])

        return self.p


class NeuroimagingSoftware(NIDMObject):
    """
    Class representing a NeuroimagingSoftware Agent.
    """

    def __init__(self, software_type, version):
        super(NeuroimagingSoftware, self).__init__()
        self.id = NIIRI[str(uuid.uuid4())]
        self.version = version
        # FIXME: get label from owl!
        if software_type.lower() == "fsl":
            self.name = "FSL"
        else:
            raise Exception('Unrecognised software: ' + str(software_type))
        self.type = SCR_FSL  # NLX_FSL
        self.prov_type = PROV['Agent']

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        if nidm_version['major'] < 1 or \
                (nidm_version['major'] == 1 and nidm_version['minor'] < 3):
            self.type = NLX_OLD_FSL

        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['type'], PROV['SoftwareAgent']),
            (PROV['label'], self.name),
            (NIDM_SOFTWARE_VERSION, self.version))
        )

        return self.p


class ExporterSoftware(NIDMObject):
    """
    Class representing a Software Agent.
    """

    def __init__(self, software_type, version):
        super(ExporterSoftware, self).__init__()
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = software_type
        self.prov_type = PROV['Agent']
        self.version = version

        if software_type == NIDM_FSL:
            self.name = "nidmfsl"
        else:
            raise Exception('Unrecognised software: ' + str(software_type))

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes((
            (PROV['type'], self.type),
            (PROV['type'], PROV['SoftwareAgent']),
            (PROV['label'], self.name),
            (NIDM_SOFTWARE_VERSION, self.version))
        )

        return self.p


class NIDMResultsExport(NIDMObject):
    """
    Class representing a NIDM-Results Export activity.
    """
    def __init__(self):
        super(NIDMResultsExport, self).__init__()
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = NIDM_NIDM_RESULTS_EXPORT
        self.prov_type = PROV['Activity']
        self.label = "NIDM-Results export"

    def export(self, nidm_version):
        """
        Create prov entities and activities.
        """
        self.add_attributes([(PROV['label'], self.label)])

        return self.p
