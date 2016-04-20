"""
Export neuroimaging results created by neuroimaging software packages
(FSL, AFNI, ...) following NIDM-Results specification.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

from nidmresults.objects.constants import *
from nidmresults.objects.modelfitting import *
from nidmresults.objects.contrast import *
from nidmresults.objects.inference import *

from rdflib.plugins.parsers.notation3 import BadSyntax
import zipfile
import csv


class Graph():
    """
    Generic class to read a NIDM-result archive and create a python object.
    """

    def __init__(self, nidm_zip=None, rdf_file=None, format="turtle"):
        with zipfile.ZipFile(nidm_zip) as z:
            rdf_data = z.read('nidm.ttl')

        self.study_name = os.path.basename(nidm_zip).replace(".nidm.zip", "")

        self.rdf_data = rdf_data

        self.format = format
        self.graph = self.parse()

        self.objects = dict()

    def parse(self):
        g = rdflib.Graph()
        try:
            g.parse(data=self.rdf_data, format=self.format)
        except BadSyntax:
            raise self.ParseException(
                "RDFLib was unable to parse the RDF file.")
        return g

    def get_statistic_maps(self):
        """
        Read a NIDM-Results document and return a list of Statistic Maps.
        """

        query = """
        prefix prov: <http://www.w3.org/ns/prov#>
        prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>
        prefix nidm_StatisticMap: <http://purl.org/nidash/nidm#NIDM_0000076>
        prefix nidm_statisticType: <http://purl.org/nidash/nidm#NIDM_0000123>
        prefix nidm_errorDegreesOfFreedom: <http://purl.org/nidash/nidm#NIDM_0\
000093>

        SELECT ?label ?contrastName ?statType ?statFile ?dof WHERE {
         ?sid a nidm_StatisticMap: ;
              nidm_contrastName: ?contrastName ;
              nidm_statisticType: ?statType ;
              rdfs:label ?label ;
              nidm_errorDegreesOfFreedom: ?dof ;
              prov:atLocation ?statFile .
        }
        """
        sd = self.graph.query(query)

        stat_maps = list()
        if sd:
            for label, contrast_name, stat_type, stat_file, dof in sd:
                contrast_num = None
                coord_space = None
                export_dir = None
                stat_maps.append(StatisticMap(
                    stat_file, stat_type, contrast_num, contrast_name, dof,
                    coord_space, export_dir, label))
        return stat_maps

    def get_peaks(self, contrast_name=None):
        """
        Read a NIDM-Results document and return a list of Peaks.
        """

        if contrast_name is not None:
            query_extension = """
?cluster a significant_cluster: .
?peak prov:wasDerivedFrom ?cluster .
?cluster prov:wasDerivedFrom/prov:wasGeneratedBy/prov:used/prov:\
wasGeneratedBy ?conest .
?conmap prov:wasGeneratedBy ?conest .
?conmap a nidm_ContrastMap: .
?conmap nidm_contrastName: ?conname .
            """
        else:
            query_extension = ""

        query = """
prefix prov: <http://www.w3.org/ns/prov#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix nidm_ExcursionSetMap: <http://purl.org/nidash/nidm#NIDM_0000025>
prefix nidm_pValueUncorrected: <http://purl.org/nidash/nidm#NIDM_0000116>
prefix nidm_equivalentZStatistic: <http://purl.org/nidash/nidm#NIDM_0000092>
prefix nidm_coordinateVector: <http://purl.org/nidash/nidm#NIDM_0000086>
prefix nidm_Coordinate: <http://purl.org/nidash/nidm#NIDM_0000015>
prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>

SELECT DISTINCT ?coord_label ?coord_vector ?z ?peak_label ?p_unc
?peak_id ?exc_set_id WHERE {
    ?coord a nidm_Coordinate: ;
        rdfs:label ?coord_label ;
        nidm_coordinateVector: ?coord_vector .
    ?peak_id prov:atLocation ?coord ;
        rdfs:label ?peak_label ;
        nidm_equivalentZStatistic: ?z ;
        nidm_pValueUncorrected: ?p_unc ;
        prov:wasDerivedFrom/prov:wasDerivedFrom ?exc_set_id .
    """ + query_extension + """
}
ORDER BY ?peak_label
        """
        sd = self.graph.query(query)

        peaks = dict()
        if sd:
            for coord_label, coord_vector, z, peak_label, p_unc, peak_id, \
                    exc_set_id in sd:
                cluster_index = None
                stat_num = None
                cluster_id = None
                local_peak_id = None
                peak = Peak(cluster_index, local_peak_id, float(z), stat_num,
                            cluster_id,
                            coord_vector=coord_vector, p_unc=float(p_unc),
                            label=peak_label, coord_label=coord_label,
                            oid=peak_id, exc_set_id=exc_set_id)
                peaks[peak_id] = (peak)

        self.objects.update(peaks)
        return peaks

    def peak_query(self):
        query = """
        prefix prov: <http://www.w3.org/ns/prov#>
        prefix spm: <http://purl.org/nidash/spm#>
        prefix nidm: <http://purl.org/nidash/nidm#>
        prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        prefix peak: <http://purl.org/nidash/nidm#NIDM_0000062>
        prefix significant_cluster: <http://purl.org/nidash/nidm#NIDM_0000070>
        prefix coordinate: <http://purl.org/nidash/nidm#NIDM_0000086>
        prefix equivalent_zstatistic: <http://purl.org/nidash/\
nidm#NIDM_0000092>
        prefix pvalue_fwer: <http://purl.org/nidash/nidm#NIDM_0000115>
        prefix pvalue_uncorrected: <http://purl.org/nidash/nidm#NIDM_0000116>
        prefix nidm_ContrastMap: <http://purl.org/nidash/nidm#NIDM_0000002>
        prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>

        ####################
        #Peaks Data Query
        ####################
        SELECT ?peak ?xyz ?cluster ?zstat ?pvalfwer ?conname
        WHERE
        { ?peak a peak: .
          ?cluster a significant_cluster: .
          ?peak prov:wasDerivedFrom ?cluster .
          ?peak prov:atLocation ?coordinate .
          ?coordinate coordinate: ?xyz .
          ?peak equivalent_zstatistic: ?zstat .
          OPTIONAL { ?peak pvalue_fwer: ?pvalfwer }.
          ?cluster prov:wasDerivedFrom/prov:wasGeneratedBy/prov:used/prov:\
wasGeneratedBy ?conest .
          ?conmap prov:wasGeneratedBy ?conest .
          ?conmap a nidm_ContrastMap: .
          ?conmap nidm_contrastName: ?conname .
        }
        ORDER BY ?cluster ?peak
        """
        sd = self.graph.query(query)

        peaks = list()
        if sd:
            for peak_id, xyz, cluster, zstat, pfwer, conname in sd:
                local_peak_id = None
                peak = Peak(None, local_peak_id, zstat, 1, cluster_id=cluster,
                            coord_vector=xyz, p_fwer=pfwer, oid=peak_id)
                peaks.append(peak)
        return peaks_df

    def get_coordinate_spaces(self, oid=None):
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
        sd = self.graph.query(query)

        coord_spaces = dict()
        if sd:
            for row in sd:
                args = row.asdict()
                if oid is not None:
                    args['oid'] = oid
                coord_spaces[args['oid']] = CoordinateSpace(**args)

        if oid is not None:
            return coord_spaces[oid]
        else:
            return coord_spaces

    def get_stat_maps(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Statistic Maps.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_StatisticMap: <http://purl.org/nidash/nidm#NIDM_0000076>
prefix nidm_statisticType: <http://purl.org/nidash/nidm#NIDM_0000123>
prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>
prefix nidm_effectDegreesOfFreedom: <http://purl.org/nidash/nidm#NIDM_0000091>
prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>
prefix nidm_errorDegreesOfFreedom: <http://purl.org/nidash/nidm#NIDM_0000093>
prefix obo_tstatistic: <http://purl.obolibrary.org/obo/STATO_0000176>

SELECT DISTINCT * WHERE {
    """ + oid_var + """ a nidm_StatisticMap: ;
        rdfs:label ?label ;
        prov:atLocation ?location ;
        nidm_statisticType: ?stat_type ;
        nfo:fileName ?filename ;
        dct:format ?format ;
        nidm_contrastName: ?contrast_name ;
        nidm_errorDegreesOfFreedom: ?dof ;
        nidm_effectDegreesOfFreedom: ?effdof ;
        nidm_inCoordinateSpace: ?coord_space_id ;
        crypto:sha512 ?sha .
}
        """
        sd = self.graph.query(query)

        stat_maps = dict()
        if sd:
            for row in sd:
                coord_space = self.get_coordinate_spaces(row.coord_space_id)
                args = row.asdict()
                args['coord_space'] = coord_space
                if oid is not None:
                    args['oid'] = oid
                # FIXME: will have to be set by default but that will change
                # position of arguments (check for compatibility)
                args['contrast_num'] = None
                args.pop("coord_space_id", None)
                stat_maps[args['oid']] = StatisticMap(**args)

        self.objects.update(stat_maps)
        if oid is not None:
            return stat_maps[oid]
        else:
            return stat_maps

    def get_inferences(self, oid=None):
        """
        Read a NIDM-Results document and return a dict of Inference activities.
        """
        if oid is None:
            oid_var = "?oid"
        else:
            oid_var = "<" + str(oid) + ">"

        query = """
prefix nidm_Inference: <http://purl.org/nidash/nidm#NIDM_0000049>
prefix nidm_hasAltHypothesis: <http://purl.org/nidash/nidm#NIDM_0000097>
prefix nidm_OneTailedTest: <http://purl.org/nidash/nidm#NIDM_0000060>
prefix nidm_StatisticMap: <http://purl.org/nidash/nidm#NIDM_0000076>

SELECT DISTINCT ?oid ?label ?tail ?stat_map_id WHERE {
    """ + oid_var + """ a nidm_Inference: ;
        rdfs:label ?label ;
        nidm_hasAltHypothesis: ?tail ;
        prov:used ?stat_map_id .
    ?stat_map_id a nidm_StatisticMap: .
}
        """
        sd = self.graph.query(query)

        inferences = dict()
        if sd:
            for row in sd:
                stat_map = self.get_stat_maps(row.stat_map_id)
                args = row.asdict()
                args['stat_map'] = stat_map
                args['contrast_num'] = None
                args.pop("stat_map_id", None)
                if oid is not None:
                    args['oid'] = oid

                inferences[args['oid']] = InferenceActivity(**args)

        self.objects.update(inferences)
        if oid is not None:
            return inferences[oid]
        else:
            return inferences

    def get_excursion_set_maps(self):
        """
        Read a NIDM-Results document and return a dict of ExcursionSet.
        """

        # We need to add optional
    #         nidm_hasClusterLabelsMap: ?cluster_label_map_id ;
    # nidm_hasMaximumIntensityProjection: ?mip_id ;
        # nidm_numberOfSignificantClusters: ?num_signif_vox ;
        # nidm_pValue: ?p_value ;
        # visu
        query = """
prefix prov: <http://www.w3.org/ns/prov#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>

prefix nidm_ExcursionSetMap: <http://purl.org/nidash/nidm#NIDM_0000025>
prefix nidm_hasClusterLabelsMap: <http://purl.org/nidash/nidm#NIDM_0000098>
prefix nidm_hasMaximumIntensityProjection: <http://purl.org/nidash/nidm#NIDM_0\
000138>
prefix nidm_inCoordinateSpace: <http://purl.org/nidash/nidm#NIDM_0000104>
prefix nidm_numberOfSignificantClusters: <http://purl.org/nidash/nidm#NIDM_000\
0111>
prefix nidm_pValue: <http://purl.org/nidash/nidm#NIDM_0000114>

SELECT DISTINCT ?oid ?label ?location ?format ?filname ?cluster_label_map_id
?mip_id ?coord_space_id ?sha ?num_signif_vox ?p_value ?inference_id WHERE {

?oid a nidm_ExcursionSetMap: ;
    prov:atLocation ?location ;
    rdfs:label ?label ;
    dct:format ?format ;
    nfo:fileName ?filename ;
    nidm_inCoordinateSpace: ?coord_space_id ;
    crypto:sha512 ?sha ;
    prov:wasGeneratedBy ?inference_id .
    OPTIONAL {?oid dc:description ?visualisation } .
}
ORDER BY ?peak_label
        """
        sd = self.graph.query(query)

        exc_sets = dict()
        if sd:
            for row in sd:
                args = row.asdict()

                coord_space = self.get_coordinate_spaces(row.coord_space_id)
                args['coord_space'] = coord_space
                args.pop("coord_space_id", None)

                inference = self.get_inferences(row.inference_id)
                args['inference'] = inference
                args.pop("inference_id", None)

                exc_sets[args['oid']] = ExcursionSet(**args)
                #             cluster_id,
                #             coord_vector=coord_vector, p_unc=float(p_unc),
                #             label=peak_label, coord_label=coord_label) #,
                #             # excursion_set_id=exc_set_id)
                # peaks[peak_id] = (peak)

        self.objects.update(exc_sets)
        return exc_sets

    def serialize(self, destination, format="mkda", overwrite=False,
                  con_ids=dict()):
        # We need the peaks, excursion set maps and contrast maps
        self.get_peaks()
        self.get_excursion_set_maps()
        self.get_inferences()
        # self.get_contrast_maps()

        if format == "mkda":
            if not destination.endswith(".csv"):
                destination = destination + ".csv"
            csvfile = destination

            if overwrite:
                add_header = True
                open_mode = 'wb'
            else:
                add_header = False
                open_mode = 'ab'

            with open(csvfile, open_mode) as fid:
                writer = csv.writer(fid, delimiter='\t')
                if add_header:
                    writer.writerow(["9", "NaN", "NaN", "NaN", "NaN", "NaN",
                                     "NaN", "NaN", "NaN"])
                    writer.writerow(
                        ["x", "y", "z", "Study", "Contrast", "N",
                         "FixedRandom", "CoordSys", "Name"])

                self.N = 20  # FIXME
                self.FixedRandom = "random"  # FIXME

                # For anything that has a label
                con_ids[None] = 0

                for oid, peak in self.get_peaks().items():
                    exc_set = self.objects[peak.exc_set_id]

                    if exc_set.coord_space.is_mni():
                        space = "MNI"
                    elif exc_set.coord_space.is_talairach():
                        space = "T88"

                    stat_map = exc_set.inference.stat_map

                    con_name = stat_map.contrast_name.replace(" ", "_")

                    if con_name in con_ids:
                        con_id = con_ids[con_name]
                    else:
                        con_id = max(con_ids.values()) + 1
                        con_ids[con_name] = con_id

                    writer.writerow([
                        peak.coordinate.coord_vector[0],
                        peak.coordinate.coord_vector[1],
                        peak.coordinate.coord_vector[2], self.study_name,
                        con_id,
                        self.N, self.FixedRandom,
                        space, con_name])

        return con_ids
