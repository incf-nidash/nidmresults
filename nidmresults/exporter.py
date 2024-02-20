"""Export neuroimaging results created by neuroimaging software packages \
(FSL, AFNI, ...) following NIDM-Results specification.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

import csv
import datetime
import json
import os
import shutil
import tempfile
import zipfile

# Needed for export using JSON-LD 1.1
import pyld as ld
from prov.model import ProvBundle, ProvDocument

from nidmresults.objects.constants import (
    CRYPTO,
    DC,
    DCT,
    NFO,
    NIDM,
    NIDM_INDEPEDENT_ERROR,
    NIF,
    NIIRI,
    OBO,
    PROV,
    SCR,
    STATO_GLS,
    STATO_OLS,
    STATO_WLS,
)
from nidmresults.objects.contrast import ContrastStdErrMap
from nidmresults.objects.generic import (
    NIDMFile,
    NIDMResultsBundle,
    NIDMResultsExport,
)
from nidmresults.objects.modelfitting import ModelParametersEstimation


class NIDMExporter:
    """Generic class to parse a result directory \
       to extract the pieces of information to be stored in NIDM-Results \
       and to generate a NIDM-Results export."""

    def __init__(self, version, out_dir, zipped=True):
        out_dirname = os.path.basename(out_dir)
        out_path = os.path.dirname(out_dir)

        # Create output path from output name
        self.zipped = zipped
        if not self.zipped:
            out_dirname = f"{out_dirname}.nidm"
        else:
            out_dirname = f"{out_dirname}.nidm.zip"
        out_dir = os.path.join(out_path, out_dirname)

        # Quit if output path already exists
        # and user doesn't want to overwrite it
        if os.path.exists(out_dir):
            msg = f"{out_dir} already exists, overwrite?"
            if input(f"{msg} (y/N) ").lower() != "y":
                quit("Bye.")
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            else:
                os.remove(out_dir)
        self.out_dir = out_dir

        if version == "dev":
            self.version = {
                "major": 10000,
                "minor": 0,
                "revision": 0,
                "num": version,
            }
        else:
            major, minor, revision = version.split(".")
            if "-rc" in revision:
                revision, rc = revision.split("-rc")
            else:
                rc = -1
            self.version = {
                "major": int(major),
                "minor": int(minor),
                "revision": int(revision),
                "rc": int(rc),
                "num": version,
            }

        # Initialise prov document
        self.doc = ProvDocument()
        self._add_namespaces()

        # A temp directory that will contain the exported data
        self.export_dir = tempfile.mkdtemp(prefix="nidm-", dir=out_path)

        self.prepend_path = ""

    def parse(self):
        """Parse a result directory to extract the pieces information \
           to be stored in NIDM-Results."""
        try:
            # Methods: find_software, find_model_fitting, find_contrasts and
            # find_inferences should be defined in the children classes and
            # return a list of NIDM Objects as specified in the objects module

            # Object of type Software describing the neuroimaging software
            # package used for the analysis
            self.software = self._find_software()

            # List of objects of type ModelFitting describing the
            # model fitting step in NIDM-Results (main activity: Model
            # Parameters Estimation)
            self.model_fittings = self._find_model_fitting()

            # Dictionary of (key, value) pairs where where key is a tuple
            # containing the identifier of a ModelParametersEstimation object
            # and a tuple of identifiers of ParameterEstimateMap objects and
            # value is an object of type Contrast describing the contrast
            # estimation step in NIDM-Results (main activity: Contrast
            # Estimation)
            self.contrasts = self._find_contrasts()

            # Inference activity and entities
            # Dictionary of (key, value) pairs where key is the identifier of a
            # ContrastEstimation object and value is an object of type
            # Inference describing the inference step in NIDM-Results (main
            # activity: Inference)
            self.inferences = self._find_inferences()
        except Exception:
            self.cleanup()
            raise

    def cleanup(self):
        if os.path.isdir(self.export_dir):
            shutil.rmtree(self.export_dir)

    def add_object(self, nidm_object, export_file=True):
        """Add a NIDMObject to a NIDM-Results export."""
        if not export_file:
            export_dir = None
        else:
            export_dir = self.export_dir

        if not isinstance(nidm_object, NIDMFile):
            nidm_object.export(self.version, export_dir)
        else:
            nidm_object.export(self.version, export_dir, self.prepend_path)
        # ProvDocument: add object to the bundle
        if nidm_object.prov_type == PROV["Activity"]:
            self.bundle.activity(
                nidm_object.id, other_attributes=nidm_object.attributes
            )
        elif nidm_object.prov_type == PROV["Entity"]:
            self.bundle.entity(
                nidm_object.id, other_attributes=nidm_object.attributes
            )
        elif nidm_object.prov_type == PROV["Agent"]:
            self.bundle.agent(
                nidm_object.id, other_attributes=nidm_object.attributes
            )
        # self.bundle.update(nidm_object.p)

    def export(self):
        """Generate a NIDM-Results export."""
        try:
            if not os.path.isdir(self.export_dir):
                os.mkdir(self.export_dir)

            # Initialise main bundle
            self._create_bundle(self.version)

            self.add_object(self.software)

            # Add model fitting steps
            if not isinstance(self.model_fittings, list):
                self.model_fittings = list(self.model_fittings.values())

            for model_fitting in self.model_fittings:
                # Design Matrix
                # model_fitting.activity.used(model_fitting.design_matrix)
                self.bundle.used(
                    model_fitting.activity.id, model_fitting.design_matrix.id
                )
                self.add_object(model_fitting.design_matrix)
                # *** Export visualisation of the design matrix
                self.add_object(model_fitting.design_matrix.image)

                if model_fitting.design_matrix.image.file is not None:
                    self.add_object(model_fitting.design_matrix.image.file)

                if model_fitting.design_matrix.hrf_models is not None:
                    # drift model
                    self.add_object(model_fitting.design_matrix.drift_model)

                if self.version["major"] > 1 or (
                    self.version["major"] == 1 and self.version["minor"] >= 3
                ):
                    # Machine
                    # model_fitting.data.wasAttributedTo(model_fitting.machine)
                    self.bundle.wasAttributedTo(
                        model_fitting.data.id, model_fitting.machine.id
                    )
                    self.add_object(model_fitting.machine)

                    # Imaged subject or group(s)
                    for sub in model_fitting.subjects:
                        self.add_object(sub)
                        # model_fitting.data.wasAttributedTo(sub)
                        self.bundle.wasAttributedTo(
                            model_fitting.data.id, sub.id
                        )

                # Data
                # model_fitting.activity.used(model_fitting.data)
                self.bundle.used(
                    model_fitting.activity.id, model_fitting.data.id
                )
                self.add_object(model_fitting.data)

                # Error Model
                # model_fitting.activity.used(model_fitting.error_model)
                self.bundle.used(
                    model_fitting.activity.id, model_fitting.error_model.id
                )
                self.add_object(model_fitting.error_model)

                # Parameter Estimate Maps
                for param_estimate in model_fitting.param_estimates:
                    # param_estimate.wasGeneratedBy(model_fitting.activity)
                    self.bundle.wasGeneratedBy(
                        param_estimate.id, model_fitting.activity.id
                    )
                    self.add_object(param_estimate)
                    self.add_object(param_estimate.coord_space)
                    self.add_object(param_estimate.file)

                    if param_estimate.derfrom is not None:
                        self.bundle.wasDerivedFrom(
                            param_estimate.id, param_estimate.derfrom.id
                        )
                        self.add_object(param_estimate.derfrom)
                        self.add_object(
                            param_estimate.derfrom.file, export_file=False
                        )

                # Residual Mean Squares Map
                # model_fitting.rms_map.wasGeneratedBy(model_fitting.activity)
                self.add_object(model_fitting.rms_map)
                self.bundle.wasGeneratedBy(
                    model_fitting.rms_map.id, model_fitting.activity.id
                )
                self.add_object(model_fitting.rms_map.coord_space)
                self.add_object(model_fitting.rms_map.file)
                if model_fitting.rms_map.derfrom is not None:
                    self.bundle.wasDerivedFrom(
                        model_fitting.rms_map.id,
                        model_fitting.rms_map.derfrom.id,
                    )
                    self.add_object(model_fitting.rms_map.derfrom)
                    self.add_object(
                        model_fitting.rms_map.derfrom.file, export_file=False
                    )

                # Resels per Voxel Map
                if model_fitting.rpv_map is not None:
                    self.add_object(model_fitting.rpv_map)
                    self.bundle.wasGeneratedBy(
                        model_fitting.rpv_map.id, model_fitting.activity.id
                    )
                    self.add_object(model_fitting.rpv_map.coord_space)
                    self.add_object(model_fitting.rpv_map.file)
                    if model_fitting.rpv_map.inf_id is not None:
                        self.bundle.used(
                            model_fitting.rpv_map.inf_id,
                            model_fitting.rpv_map.id,
                        )
                    if model_fitting.rpv_map.derfrom is not None:
                        self.bundle.wasDerivedFrom(
                            model_fitting.rpv_map.id,
                            model_fitting.rpv_map.derfrom.id,
                        )
                        self.add_object(model_fitting.rpv_map.derfrom)
                        self.add_object(
                            model_fitting.rpv_map.derfrom.file,
                            export_file=False,
                        )

                # Mask
                # model_fitting.mask_map.wasGeneratedBy(model_fitting.activity)
                self.bundle.wasGeneratedBy(
                    model_fitting.mask_map.id, model_fitting.activity.id
                )
                self.add_object(model_fitting.mask_map)
                if model_fitting.mask_map.derfrom is not None:
                    self.bundle.wasDerivedFrom(
                        model_fitting.mask_map.id,
                        model_fitting.mask_map.derfrom.id,
                    )
                    self.add_object(model_fitting.mask_map.derfrom)
                    self.add_object(
                        model_fitting.mask_map.derfrom.file, export_file=False
                    )

                # Create coordinate space export
                self.add_object(model_fitting.mask_map.coord_space)
                # Create "Mask map" entity
                self.add_object(model_fitting.mask_map.file)

                # Grand Mean map
                # model_fitting.grand_mean_map.wasGeneratedBy(model_fitting.activity)
                self.bundle.wasGeneratedBy(
                    model_fitting.grand_mean_map.id, model_fitting.activity.id
                )
                self.add_object(model_fitting.grand_mean_map)
                # Coordinate space entity
                self.add_object(model_fitting.grand_mean_map.coord_space)
                # Grand Mean Map entity
                self.add_object(model_fitting.grand_mean_map.file)

                # Model Parameters Estimation activity
                self.add_object(model_fitting.activity)
                self.bundle.wasAssociatedWith(
                    model_fitting.activity.id, self.software.id
                )
                # model_fitting.activity.wasAssociatedWith(self.software)
                # self.add_object(model_fitting)

            # Add contrast estimation steps
            analysis_masks = {}
            for (model_fitting_id, pe_ids), contrasts in list(
                self.contrasts.items()
            ):
                for contrast in contrasts:
                    model_fitting = self._get_model_fitting(model_fitting_id)
                    # for contrast in contrasts:
                    # contrast.estimation.used(model_fitting.rms_map)
                    self.bundle.used(
                        contrast.estimation.id, model_fitting.rms_map.id
                    )
                    # contrast.estimation.used(model_fitting.mask_map)
                    self.bundle.used(
                        contrast.estimation.id, model_fitting.mask_map.id
                    )
                    analysis_masks[contrast.estimation.id] = (
                        model_fitting.mask_map.id
                    )
                    self.bundle.used(
                        contrast.estimation.id, contrast.weights.id
                    )
                    self.bundle.used(
                        contrast.estimation.id, model_fitting.design_matrix.id
                    )
                    # contrast.estimation.wasAssociatedWith(self.software)
                    self.bundle.wasAssociatedWith(
                        contrast.estimation.id, self.software.id
                    )

                    for pe_id in pe_ids:
                        # contrast.estimation.used(pe_id)
                        self.bundle.used(contrast.estimation.id, pe_id)

                    # Create estimation activity
                    self.add_object(contrast.estimation)

                    # Create contrast weights
                    self.add_object(contrast.weights)

                    if contrast.contrast_map is not None:
                        # Create contrast Map
                        # contrast.contrast_map.wasGeneratedBy(contrast.estimation)
                        self.bundle.wasGeneratedBy(
                            contrast.contrast_map.id, contrast.estimation.id
                        )
                        self.add_object(contrast.contrast_map)
                        self.add_object(contrast.contrast_map.coord_space)
                        # Copy contrast map in export directory
                        self.add_object(contrast.contrast_map.file)

                        if contrast.contrast_map.derfrom is not None:
                            self.bundle.wasDerivedFrom(
                                contrast.contrast_map.id,
                                contrast.contrast_map.derfrom.id,
                            )
                            self.add_object(contrast.contrast_map.derfrom)
                            self.add_object(
                                contrast.contrast_map.derfrom.file,
                                export_file=False,
                            )

                    # Create Std Err. Map (T-tests) or Explained Mean Sq. Map
                    # (F-tests)
                    # contrast.stderr_or_expl_mean_sq_map.wasGeneratedBy
                    # (contrast.estimation)
                    stderr_explmeansq_map = contrast.stderr_or_expl_mean_sq_map
                    self.bundle.wasGeneratedBy(
                        stderr_explmeansq_map.id, contrast.estimation.id
                    )
                    self.add_object(stderr_explmeansq_map)
                    self.add_object(stderr_explmeansq_map.coord_space)
                    if (
                        isinstance(stderr_explmeansq_map, ContrastStdErrMap)
                        and stderr_explmeansq_map.contrast_var
                    ):
                        self.add_object(stderr_explmeansq_map.contrast_var)
                        if stderr_explmeansq_map.var_coord_space:
                            self.add_object(
                                stderr_explmeansq_map.var_coord_space
                            )
                        if stderr_explmeansq_map.contrast_var.coord_space:
                            self.add_object(
                                stderr_explmeansq_map.contrast_var.coord_space
                            )
                        self.add_object(
                            stderr_explmeansq_map.contrast_var.file,
                            export_file=False,
                        )
                        self.bundle.wasDerivedFrom(
                            stderr_explmeansq_map.id,
                            stderr_explmeansq_map.contrast_var.id,
                        )
                    self.add_object(stderr_explmeansq_map.file)

                    # Create Statistic Map
                    # contrast.stat_map.wasGeneratedBy(contrast.estimation)
                    self.bundle.wasGeneratedBy(
                        contrast.stat_map.id, contrast.estimation.id
                    )
                    self.add_object(contrast.stat_map)
                    self.add_object(contrast.stat_map.coord_space)
                    # Copy Statistical map in export directory
                    self.add_object(contrast.stat_map.file)

                    if contrast.stat_map.derfrom is not None:
                        self.bundle.wasDerivedFrom(
                            contrast.stat_map.id, contrast.stat_map.derfrom.id
                        )
                        self.add_object(contrast.stat_map.derfrom)
                        self.add_object(
                            contrast.stat_map.derfrom.file, export_file=False
                        )

                    # Create Z Statistic Map
                    if contrast.z_stat_map:
                        # contrast.z_stat_map.wasGeneratedBy(contrast.estimation)
                        self.bundle.wasGeneratedBy(
                            contrast.z_stat_map.id, contrast.estimation.id
                        )
                        self.add_object(contrast.z_stat_map)
                        self.add_object(contrast.z_stat_map.coord_space)
                        # Copy Statistical map in export directory
                        self.add_object(contrast.z_stat_map.file)

                    # self.add_object(contrast)

            # Add inference steps
            for contrast_id, inferences in list(self.inferences.items()):
                contrast = self._get_contrast(contrast_id)

                for inference in inferences:
                    if contrast.z_stat_map:
                        used_id = contrast.z_stat_map.id
                    else:
                        used_id = contrast.stat_map.id
                    # inference.inference_act.used(used_id)
                    self.bundle.used(inference.inference_act.id, used_id)
                    # inference.inference_act.wasAssociatedWith(self.software)
                    self.bundle.wasAssociatedWith(
                        inference.inference_act.id, self.software.id
                    )

                    # self.add_object(inference)
                    # Excursion set
                    # inference.excursion_set.wasGeneratedBy(inference.inference_act)
                    self.bundle.wasGeneratedBy(
                        inference.excursion_set.id, inference.inference_act.id
                    )
                    self.add_object(inference.excursion_set)
                    self.add_object(inference.excursion_set.coord_space)
                    if inference.excursion_set.visu is not None:
                        self.add_object(inference.excursion_set.visu)
                        if inference.excursion_set.visu.file is not None:
                            self.add_object(inference.excursion_set.visu.file)
                    # Copy "Excursion set map" file in export directory
                    self.add_object(inference.excursion_set.file)
                    if inference.excursion_set.clust_map is not None:
                        self.add_object(inference.excursion_set.clust_map)
                        self.add_object(inference.excursion_set.clust_map.file)
                        self.add_object(
                            inference.excursion_set.clust_map.coord_space
                        )

                    if inference.excursion_set.mip is not None:
                        self.add_object(inference.excursion_set.mip)
                        self.add_object(inference.excursion_set.mip.file)

                    # Height threshold
                    if inference.height_thresh.equiv_thresh is not None:
                        for equiv in inference.height_thresh.equiv_thresh:
                            self.add_object(equiv)
                    self.add_object(inference.height_thresh)

                    # Extent threshold
                    if inference.extent_thresh.equiv_thresh is not None:
                        for equiv in inference.extent_thresh.equiv_thresh:
                            self.add_object(equiv)
                    self.add_object(inference.extent_thresh)

                    # Display Mask (potentially more than 1)
                    if inference.disp_mask:
                        for mask in inference.disp_mask:
                            # inference.inference_act.used(mask)
                            self.bundle.used(
                                inference.inference_act.id, mask.id
                            )
                            self.add_object(mask)
                            # Create coordinate space entity
                            self.add_object(mask.coord_space)
                            # Create "Display Mask Map" entity
                            self.add_object(mask.file)

                            if mask.derfrom is not None:
                                self.bundle.wasDerivedFrom(
                                    mask.id, mask.derfrom.id
                                )
                                self.add_object(mask.derfrom)
                                self.add_object(
                                    mask.derfrom.file, export_file=False
                                )

                    # Search Space
                    self.bundle.wasGeneratedBy(
                        inference.search_space.id, inference.inference_act.id
                    )
                    # inference.search_space.wasGeneratedBy(inference.inference_act)
                    self.add_object(inference.search_space)
                    self.add_object(inference.search_space.coord_space)
                    # Copy "Mask map" in export directory
                    self.add_object(inference.search_space.file)

                    # Peak Definition
                    if inference.peak_criteria:
                        # inference.inference_act.used(inference.peak_criteria)
                        self.bundle.used(
                            inference.inference_act.id,
                            inference.peak_criteria.id,
                        )
                        self.add_object(inference.peak_criteria)

                    # Cluster Definition
                    if inference.cluster_criteria:
                        # inference.inference_act.used(inference.cluster_criteria)
                        self.bundle.used(
                            inference.inference_act.id,
                            inference.cluster_criteria.id,
                        )
                        self.add_object(inference.cluster_criteria)

                    if inference.clusters:
                        # Clusters and peaks
                        for cluster in inference.clusters:
                            # cluster.wasDerivedFrom(inference.excursion_set)
                            self.bundle.wasDerivedFrom(
                                cluster.id, inference.excursion_set.id
                            )
                            self.add_object(cluster)
                            for peak in cluster.peaks:
                                self.bundle.wasDerivedFrom(peak.id, cluster.id)
                                self.add_object(peak)
                                self.add_object(peak.coordinate)

                            if cluster.cog is not None:
                                self.bundle.wasDerivedFrom(
                                    cluster.cog.id, cluster.id
                                )
                                self.add_object(cluster.cog)
                                self.add_object(cluster.cog.coordinate)

                    # Inference activity
                    # inference.inference_act.wasAssociatedWith(inference.software_id)
                    # inference.inference_act.used(inference.height_thresh)
                    self.bundle.used(
                        inference.inference_act.id, inference.height_thresh.id
                    )
                    # inference.inference_act.used(inference.extent_thresh)
                    self.bundle.used(
                        inference.inference_act.id, inference.extent_thresh.id
                    )
                    self.bundle.used(
                        inference.inference_act.id,
                        analysis_masks[contrast.estimation.id],
                    )
                    self.add_object(inference.inference_act)

            # Write-out prov file
            self.save_prov_to_files()

            return self.out_dir
        except Exception:
            self.cleanup()
            raise

    def _get_model_fitting(self, mf_id):
        """Retrieve model fitting with identifier 'mf_id' \
           from the list of model fitting objects stored \
           in self.model_fitting."""
        for model_fitting in self.model_fittings:
            if model_fitting.activity.id == mf_id:
                return model_fitting

        raise Exception(
            f"Model fitting activity with id: {str(mf_id)} not found."
        )

    def _get_contrast(self, con_id):
        """Retrieve contrast with identifier 'con_id' \
           from the list of contrast objects stored \
           in self.contrasts."""
        for contrasts in list(self.contrasts.values()):
            for contrast in contrasts:
                if contrast.estimation.id == con_id:
                    return contrast
        raise Exception(f"Contrast activity with id: {str(con_id)} not found.")

    def _add_namespaces(self):
        """Add namespaces to NIDM document."""
        self.doc.add_namespace(NIDM)
        self.doc.add_namespace(NIIRI)
        self.doc.add_namespace(CRYPTO)
        self.doc.add_namespace(DCT)
        self.doc.add_namespace(DC)
        self.doc.add_namespace(NFO)
        self.doc.add_namespace(OBO)
        self.doc.add_namespace(SCR)
        self.doc.add_namespace(NIF)

    def _create_bundle(self, version):
        """Initialise NIDM-Results bundle."""
        # *** Bundle entity
        if not hasattr(self, "bundle_ent"):
            self.bundle_ent = NIDMResultsBundle(nidm_version=version["num"])

        self.bundle = ProvBundle(identifier=self.bundle_ent.id)

        self.bundle_ent.export(self.version, self.export_dir)

        # # provn export
        # self.bundle = ProvBundle(identifier=bundle_id)

        self.doc.entity(
            self.bundle_ent.id, other_attributes=self.bundle_ent.attributes
        )

        # *** NIDM-Results Export Activity
        if version["num"] not in ["1.0.0", "1.1.0"]:
            if not hasattr(self, "export_act"):
                self.export_act = NIDMResultsExport()
            self.export_act.export(self.version, self.export_dir)
            # self.doc.update(self.export_act.p)
            self.doc.activity(
                self.export_act.id, other_attributes=self.export_act.attributes
            )

        # *** bundle was Generated by NIDM-Results Export Activity
        if not hasattr(self, "export_time"):
            self.export_time = str(datetime.datetime.now().time())

        if version["num"] in ["1.0.0", "1.1.0"]:
            self.doc.wasGeneratedBy(
                entity=self.bundle_ent.id, time=self.export_time
            )
        else:
            # provn
            self.doc.wasGeneratedBy(
                entity=self.bundle_ent.id,
                activity=self.export_act.id,
                time=self.export_time,
            )

        # *** NIDM-Results Exporter (Software Agent)
        if version["num"] not in ["1.0.0", "1.1.0"]:
            if not hasattr(self, "exporter"):
                self.exporter = self._get_exporter()
            self.exporter.export(self.version, self.export_dir)
            # self.doc.update(self.exporter.p)
            self.doc.agent(
                self.exporter.id, other_attributes=self.exporter.attributes
            )

            self.doc.wasAssociatedWith(self.export_act.id, self.exporter.id)

    def _get_model_parameters_estimations(self, error_model):
        """Infer model estimation method from the 'error_model'.

        Return an object of type ModelParametersEstimation.
        """
        if error_model.dependence == NIDM_INDEPEDENT_ERROR:
            if error_model.variance_homo:
                estimation_method = STATO_OLS
            else:
                estimation_method = STATO_WLS
        else:
            estimation_method = STATO_GLS

        mpe = ModelParametersEstimation(estimation_method, self.software.id)

        return mpe

    def use_prefixes(self, ttl):
        prefix_file = os.path.join(os.path.dirname(__file__), "prefixes.csv")
        context = {}
        with open(prefix_file, encoding="ascii") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip the headers
            for alphanum_id, prefix, uri in reader:
                if alphanum_id in ttl:
                    context[prefix] = uri
                    ttl = f"@prefix {prefix}: <{uri}" + "> .\n" + ttl
                    ttl = ttl.replace(alphanum_id, f"{prefix}:")
                    if uri in ttl:
                        ttl = ttl.replace(alphanum_id, f"{prefix}:")
                elif uri in ttl:
                    context[prefix] = uri
                    ttl = f"@prefix {prefix}: <{uri}" + "> .\n" + ttl
                    ttl = ttl.replace(alphanum_id, f"{prefix}:")
        return (ttl, context)

    def save_prov_to_files(self, showattributes=False):
        """Write-out provn serialisation to nidm.provn."""
        self.doc.add_bundle(self.bundle)
        # provn_file = os.path.join(self.export_dir, 'nidm.provn')
        # provn_fid = open(provn_file, 'w')
        # # FIXME None
        # # provn_fid.write(self.doc.get_provn(4).replace("None", "-"))
        # provn_fid.close()

        ttl_file = os.path.join(self.export_dir, "nidm.ttl")
        ttl_txt = self.doc.serialize(format="rdf", rdf_format="turtle")
        ttl_txt, json_context = self.use_prefixes(ttl_txt)

        # Add namespaces to json-ld context
        for namespace in self.doc._namespaces.get_registered_namespaces():
            json_context[namespace._prefix] = namespace._uri
        for namespace in list(
            self.doc._namespaces._default_namespaces.values()
        ):
            json_context[namespace._prefix] = namespace._uri
        json_context["xsd"] = "http://www.w3.org/2000/01/rdf-schema#"

        # Work-around to issue with INF value in rdflib (reported in
        # https://github.com/RDFLib/rdflib/pull/655)
        ttl_txt = ttl_txt.replace(" inf ", ' "INF"^^xsd:float ')
        with open(ttl_file, "w") as ttl_fid:
            ttl_fid.write(ttl_txt)

        # JSON-LD (deprecated kept for background compatibility w/ viewers)
        jsonld_file = os.path.join(self.export_dir, "nidm_deprecated.json")
        jsonld_txt = self.doc.serialize(
            format="rdf", rdf_format="json-ld", context=json_context
        )
        with open(jsonld_file, "w") as jsonld_fid:
            jsonld_fid.write(jsonld_txt)

        # JSON-LD using specification 1.1 (a.k.a "nice" JSON-LD)
        jsonld_11 = json.dumps(
            ld.jsonld.compact(
                json.loads(jsonld_txt), "http://purl.org/nidash/context"
            )
        )

        jsonld_11_file = os.path.join(self.export_dir, "nidm.json")

        with open(jsonld_11_file, "w") as fid:
            fid.write(jsonld_11)

        # provjsonld_file = os.path.join(self.export_dir, 'nidm.provjsonld')
        # provjsonld_txt = self.doc.serialize(format='jsonld')
        # with open(provjsonld_file, 'w') as provjsonld_fid:
        #     provjsonld_fid.write(provjsonld_txt)

        # provn_file = os.path.join(self.export_dir, 'nidm.provn')
        # provn_txt = self.doc.serialize(format='provn')
        # with open(provn_file, 'w') as provn_fid:
        #     provn_fid.write(provn_txt)

        # Post-processing
        if not self.zipped:
            # Just rename temp directory to output_path
            os.rename(self.export_dir, self.out_dir)
        else:
            # Create a zip file that contains the content of the temp directory
            os.chdir(self.export_dir)
            zf = zipfile.ZipFile(os.path.join("..", self.out_dir), mode="w")
            try:
                for root, dirnames, filenames in os.walk("."):
                    for filename in filenames:
                        zf.write(os.path.join(filename))
            finally:
                zf.close()
                # Need to move up before deleting the folder
                os.chdir("..")
                shutil.rmtree(os.path.join("..", self.export_dir))

        # ttl_fid = open(ttl_file, 'w');
        # serialization is done in xlm rdf
        # graph = Graph()
        # graph.parse(data=self.doc.serialize(format='rdf'), format="xml")
        # ttl_fid.write(graph.serialize(format="turtle"))
        # ttl_fid.write(self.doc.serialize(format='rdf').
        # replace("inf", '"INF"'))
        # ttl_fid.close()
        # print("provconvert -infile " + provn_file + " -outfile " + ttl_file)
        # check_call("provconvert -infile " + provn_file +
        #            " -outfile " + ttl_file, shell=True)
