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

    def __init__(self, export_dir=None):
        self.export_dir = export_dir
        self.p = ProvBundle()

        self.g = rdflib.Graph()
        self.id = None

    def copy_nifti(self, original_file, new_file):
        shutil.copy(original_file, new_file)
        path, new_filename = os.path.split(new_file)
        path, original_filename = os.path.split(original_file)

        return original_filename, new_filename

    def get_sha_sum(self, nifti_file):
        nifti_img = nib.load(nifti_file)
        data = nifti_img.get_data()
        # Fix needed as in https://github.com/pymc-devs/pymc/issues/327
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        return hashlib.sha512(data).hexdigest()

    def _rdf_add_attributes(self, attributes):
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
            if isinstance(o, str) and not str(p) == "prov:label":
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

    def add_object(self, nidm_object):
        nidm_object.export()
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
        else:
            raise Exception('Unrecognised prov relation')

        self._rdf_add(self.id, relation, object_id)

    def add_attributes(self, attributes):
        if self.type == PROV['Activity']:
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

    def __init__(self, coordinate_system, nifti_file):
        super(CoordinateSpace, self).__init__()
        self.coordinate_system = coordinate_system
        self.id = NIIRI[str(uuid.uuid4())]
        self.type = NIDM_COORDINATE_SPACE
        self.prov_type = PROV['Entity']

        thresImg = nib.load(nifti_file)
        thresImgHdr = thresImg.get_header()

        self.number_of_dimensions = len(thresImg.shape)

        self.dimension = str(thresImg.shape).replace(
            '(', '[ ').replace(')', ' ]')
        self.voxel_to_world = '%s' \
            % ', '.join(str(thresImg.get_qform())
                        .strip('()')
                        .replace('. ', '')
                        .split()).replace('[,', '[').replace('\n', '')
        self.voxel_size = '[ %s ]' % ', '.join(
            map(str, thresImgHdr['pixdim'][1:(self.number_of_dimensions + 1)]))

    def export(self):
        """
        Create prov entities and activities.
        """
        self.add_attributes({
            PROV['type']: self.type,
            NIDM_DIMENSIONS_IN_VOXELS: self.dimension,
            NIDM_NUMBER_OF_DIMENSIONS: self.number_of_dimensions,
            NIDM_VOXEL_TO_WORLD_MAPPING: self.voxel_to_world,
            NIDM_IN_WORLD_COORDINATE_SYSTEM: self.coordinate_system,
            # FIXME: this gives mm, sec => what is wrong: FSL file, nibabel,
            # other?
            # NIDM_VOXEL_UNITS:
            # '[%s]'%str(thresImgHdr.get_xyzt_units()).strip('()'),
            NIDM_VOXEL_UNITS: json.dumps(["mm", "mm", "mm"]),
            NIDM_VOXEL_SIZE: self.voxel_size,
            PROV['label']: "Coordinate space"})
        return self.p


class Image(NIDMObject):

    """
    Object representing an Image entity.
    """

    def __init__(self, export_dir, image_file):
        super(Image, self).__init__(export_dir)
        self.type = DCTYPE['Image']
        self.prov_type = PROV['Entity']
        self.file = image_file
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov entity.
        """
        if self.file is not None:
            # FIXME: replace by another name
            new_file = os.path.join(self.export_dir, "DesignMatrix.png")
            orig_filename, filename = self.copy_nifti(self.file, new_file)

            self.add_attributes([
                (PROV['type'], self.type),
                (PROV['atLocation'], Identifier("file://./" + filename)),
                (NFO['fileName'], orig_filename),
                (NFO['fileName'], filename),
                (DCT['format'], "image/png"),
            ])

        return self.p
