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

        self.objects = list()

        # self.contrast_query()
        # The peaks are the entry point
        # self.get_peaks()
        # print self.objects
        # self.statistic_maps = self.get_statistic_maps()
        # print self.contrasts
        # self.peak_query()

        # for peak in self.peaks:
        #     print self.contrasts[peak.cluster].name

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
        nidm_pValueUncorrected: ?p_unc .
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
                # print peak_id
                # FIXME: need to pass peak index!!!
                peak = Peak(cluster_index, local_peak_id, float(z), stat_num,
                            cluster_id,
                            coord_vector=coord_vector, p_unc=float(p_unc),
                            label=peak_label, coord_label=coord_label,
                            oid=peak_id) #,
                            # excursion_set_id=exc_set_id)
                peaks[peak_id] = (peak)
                print peaks

        self.objects.append(peaks)
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
                            coord_vector=xyz, p_fwer=pfwer)
                peaks.append(peak)
        return peaks_df

    def get_excursion_set_maps(self):
        """
        Read a NIDM-Results document and return a dict of ExcursionSet.
        """
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

SELECT DISTINCT ?id ?label ?loc ?format ?filname ?cluster_label_map_id ?mip_id
?coord_space_id ?sha ?num_signif_vox ?p_value ?inference_id WHERE {

?id a nidm_ExcursionSetMap: ;
    prov:atLocation ?loc ;
    rdfs:label ?label ;
    dct:format ?format ;
    nfo:fileName ?filename ;
    nidm_hasClusterLabelsMap: ?cluster_label_map_id ;
    nidm_hasMaximumIntensityProjection: ?mip_id ;
    nidm_inCoordinateSpace: ?coord_space_id ;
    crypto:sha512 ?sha ;
    nidm_numberOfSignificantClusters: ?num_signif_vox ;
    nidm_pValue: ?p_value ;
    prov:wasGeneratedBy ?inference_id .
}
ORDER BY ?peak_label
        """
        sd = self.graph.query(query)

        exc_sets = dict()
        if sd:
            for eid, label, loc, format, filname, clusterlabelmap_id, mip_id,\
                    coord_space_id, sha, num_signif_vox, p_value, inference_id\
                    in sd:
                exc_set = 1
                # cluster_index = None
                # stat_num = None
                # cluster_id = None
                # print peak_id
                exc_set = ExcursionSet(exc_file, stat_num, visualisation, coord_space,
                 export_dir)
                #             cluster_id,
                #             coord_vector=coord_vector, p_unc=float(p_unc),
                #             label=peak_label, coord_label=coord_label) #,
                #             # excursion_set_id=exc_set_id)
                # peaks[peak_id] = (peak)

        self.objects.append(exc_sets)
        return exc_sets

    def serialize(self, destination, format="mkda"):
        # We need the peaks, excursion set maps and contrast maps
        self.get_peaks()
        self.get_excursion_set_maps()
        # self.get_contrast_maps()

        if format == "mkda":
            if not destination.endswith(".csv"):
                destination = destination + ".csv"
            csvfile = destination
            with open(csvfile, 'wb') as fid:
                writer = csv.writer(fid, delimiter='\t')
                writer.writerow(["8"])
                writer.writerow(
                    ["x", "y", "z", "Study", "Contrast", "N",
                     "FixedRandom", "CoordSys"])

                self.N = 20  # FIXME
                self.FixedRandom = "random"  # FIXME

                if self.is_mni:
                    space = "MNI"
                elif self.is_talairach:
                    space = "T88"

                # For anything that has a label
                for peak in self.get_peaks():
                    writer.writerow([
                        peak.x, peak.y, peak.z, self.study_name,
                        peak.get_contrast.name(), self.N, self.FixedRandom,
                        space])

    #     # print pd.concat(peaks_df)
    # def contrast_query(self):
    #     query = """
    #     prefix prov: <http://www.w3.org/ns/prov#>
    #     prefix nidm: <http://purl.org/nidash/nidm#>

    #     prefix contrast_estimation: <http://purl.org/nidash/\
    # nidm#NIDM_0000001>
    #     prefix contrast_map: <http://purl.org/nidash/nidm#NIDM_0000002>
    #     prefix nidm_contrastName: <http://purl.org/nidash/nidm#NIDM_0000085>
    #     prefix nidm_StatisticMap:: <http://purl.org/nidash/nidm#NIDM_0000076>
    #     prefix nidm_statisticType: <http://purl.org/nidash/nidm#NIDM_0000123>

    #     SELECT ?cid ?contrastName
    #     WHERE {
    #      ?cid a contrast_map: ;
    #           nidm_contrastName: ?contrastName .
    #      ?cea a contrast_estimation: .
    #      ?cid prov:wasGeneratedBy ?cea .
    #      ?sid a nidm_StatisticMap:: ;
    #           nidm_statisticType: ?statType ;
    #           prov:atLocation ?statFile .
    #     }
    #     """
    #     sd = self.g.query(query)

    #     self.contrasts = dict()
    #     if sd:
    #         for con_id, con_name in sd:
    #             self.contrasts[con_id] = ContrastMap(
    #                 None, None, con_name, None, None, ident=con_id)



    #             print self.contrasts
    #             # peaks_df.append(peak.dataframe())
    #             # print row

    #     # print pd.concat(peaks_df)
