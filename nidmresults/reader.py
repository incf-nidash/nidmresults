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


class NIDMReader():
    """
    Generic class to read a NIDM-result archive and create a python object.
    """

    def __init__(self, rdf_file, format="turtle"):
        # self.contrast_query()
        self.get_peaks(rdf_file, format)
        # print self.contrasts
        # self.peak_query()

        # for peak in self.peaks:
        #     print self.contrasts[peak.cluster].name

    def load_graph(self, rdf_file, format="turtle"):
        g = rdflib.Graph()
        try:
            g.parse(rdf_file, format=format)
        except BadSyntax:
            raise self.ParseException(
                "RDFLib was unable to parse the RDF file.")
        return g

    def get_statistic_maps(self, rdf_file, format="turtle"):
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
        g = self.load_graph(rdf_file)
        sd = g.query(query)

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

    def get_peaks(self, rdf_file, format="turtle"):
        """
        Read a NIDM-Results document and return a list of Peaks.
        """
        query = """
        prefix prov: <http://www.w3.org/ns/prov#>
        prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        prefix nidm_pValueUncorrected: <http://purl.org/nidash/nidm#NIDM_00001\
16>
        prefix nidm_equivalentZStatistic: <http://purl.org/nidash/nidm#NIDM_00\
00092>
        prefix nidm_coordinateVector: <http://purl.org/nidash/nidm#NIDM_000008\
6>
        prefix nidm_Coordinate: <http://purl.org/nidash/nidm#NIDM_0000015>

        SELECT DISTINCT ?coord_label ?coord_vector ?z ?peak_label ?p_unc
        ?peak_id WHERE {
            ?coord a nidm_Coordinate: ;
                rdfs:label ?coord_label ;
                nidm_coordinateVector: ?coord_vector .
            ?peak prov:atLocation ?coord ;
                rdfs:label ?peak_label ;
                nidm_equivalentZStatistic: ?z ;
                nidm_pValueUncorrected: ?pvalue_uncorrected .
        }
        ORDER BY ?peak_label
        """
        g = self.load_graph(rdf_file)
        sd = g.query(query)

        peaks = list()
        if sd:
            for coord_label, coord_vector, z, peak_label, p_unc, peak_id in sd:
                cluster_index = None
                stat_num = None
                cluster_id = None
                peak = Peak(cluster_index, peak_id, z, stat_num, cluster_id,
                            coord_vector=coord_vector, p_unc=p_unc,
                            label=peak_label)
                peaks.append(peak)
        print peaks
        return peaks

    def peak_query(self, rdf_file, format="turtle"):
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
        g = self.load_graph(rdf_file)
        sd = g.query(query)

        peaks = list()
        if sd:
            for peak_id, xyz, cluster, zstat, pfwer, conname in sd:
                peak = Peak(None, peak_id, zstat, 1, cluster_id=cluster,
                            coord_vector=xyz, p_fwer=pfwer)
                peaks.append(peak)
        return peaks_df

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
