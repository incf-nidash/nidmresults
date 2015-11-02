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
from pandas import DataFrame
import pandas as pd


class NIDMReader():
    """
    Generic class to read a NIDM-result archive and create a python object.
    """

    def __init__(self, rdf_file, format="turtle"):
        g = rdflib.Graph()
        g.parse(rdf_file, format=format)
        self.g = g

        # self.contrast_query()
        self.peak_query()

        # for peak in self.peaks:
        #     print self.contrasts[peak.cluster].name

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
        sd = self.g.query(query)

        peaks_df = pd.DataFrame()
        if sd:
            for peak_id, xyz, cluster, zstat, pfwer, conname in sd:
                peak = Peak(None, peak_id, zstat, 1, cluster_id=cluster,
                            coord_vector=xyz, p_fwer=pfwer)
                peaks_df = peaks_df.append(
                    pd.concat([peak.dataframe(),
                               DataFrame({'conname': [conname]})], axis=1))

        print peaks_df
        return peaks_df

    #     # print pd.concat(peaks_df)
    # def contrast_query(self):
    #     query = """
    #     prefix prov: <http://www.w3.org/ns/prov#>
    #     prefix nidm: <http://purl.org/nidash/nidm#>

    #     prefix contrast_estimation: <http://purl.org/nidash/\
    # nidm#NIDM_0000001>
    #     prefix contrast_map: <http://purl.org/nidash/nidm#NIDM_0000002>
    #     prefix contrast_name: <http://purl.org/nidash/nidm#NIDM_0000085>
    #     prefix statistic_map: <http://purl.org/nidash/nidm#NIDM_0000076>
    #     prefix statistic_type: <http://purl.org/nidash/nidm#NIDM_0000123>

    #     SELECT ?cid ?contrastName
    #     WHERE {
    #      ?cid a contrast_map: ;
    #           contrast_name: ?contrastName .
    #      ?cea a contrast_estimation: .
    #      ?cid prov:wasGeneratedBy ?cea .
    #      ?sid a statistic_map: ;
    #           statistic_type: ?statType ;
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
