import nidmresults as nres

# Reading an existing NIDM-Results
nidm = nres.load('/Users/cmaumet/Projects/Data_sharing/dev/nidmresults_dev/test/data/ex_spm_thr_clustfwep05.nidm.zip')

nidm_meta = nidm.get_metadata()
print(nidm_meta)
# print(nidm_meta['nidm_Data/nidm_hasMRIProtocol'])

# print(dir(nidm.peaks.itervalues().next()))
# # print(dir(nidm))
# # print(nidm_meta)

# # Writing a NIDM-Results pack from a JSON file
# nidm = nres.load('/Users/cmaumet/ex_spm_thr_clustfwep05.min_nidm.json')
# nidm.save('/Users/cmaumet/ex_spm_thr_clustfwep05.min_nidm.nidm.zip')
