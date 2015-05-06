"""
Generic objects supporting the classes defined in NIDM-Results.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""
from prov.model import ProvBundle, Identifier
import numpy as np
import os
from constants import *
import nibabel as nib
import shutil
import hashlib
import uuid


class NIDMObject(object):

    """
    Generic class, parent of all objects describing a NIDM entity, activity
    or agent
    """

    def __init__(self, export_dir=None):
        self.export_dir = export_dir
        self.p = ProvBundle()
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


class CoordinateSpace(NIDMObject):

    """
    Object representing a CoordinateSpace entity.
    """

    def __init__(self, coordinate_system, nifti_file):
        super(CoordinateSpace, self).__init__()
        self.coordinate_system = coordinate_system
        self.id = NIIRI[str(uuid.uuid4())]

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
        self.p.entity(self.id, other_attributes={
            PROV['type']: NIDM_COORDINATE_SPACE,
            NIDM_DIMENSIONS_IN_VOXELS: self.dimension,
            NIDM_NUMBER_OF_DIMENSIONS: self.number_of_dimensions,
            NIDM_VOXEL_TO_WORLD_MAPPING: self.voxel_to_world,
            NIDM_IN_WORLD_COORDINATE_SYSTEM: self.coordinate_system,
            # FIXME: this gives mm, sec => what is wrong: FSL file, nibabel,
            # other?
            # NIDM_VOXEL_UNITS:
            # '[%s]'%str(thresImgHdr.get_xyzt_units()).strip('()'),
            NIDM_VOXEL_UNITS: "[ 'mm', 'mm', 'mm' ]",
            NIDM_VOXEL_SIZE: self.voxel_size,
            PROV['label']: "Coordinate space"})
        return self.p


class Image(NIDMObject):

    """
    Object representing an Image entity.
    """

    def __init__(self, export_dir, image_file):
        super(Image, self).__init__(export_dir)
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

            self.p.entity(self.id, other_attributes={
                PROV['type']: DCTYPE['Image'],
                PROV['atLocation']: Identifier("file://./" + filename),
                NFO['fileName']: orig_filename,
                NFO['fileName']: filename,
                DCT['format']: "image/png",
            })

        return self.p
