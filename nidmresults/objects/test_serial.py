from prov.model import ProvDocument, Namespace, Literal, PROV, Identifier

g = ProvDocument()
g.add_namespace('chairs', 'https://lists.w3.org/Archives/Member/chairs/')
g.add_namespace('trans', 'http://www.w3.org/2005/08/01-transitions.html#')


g.entity('chairs:2011OctDec/0004', {'prov:type': 'trans:transreq'})

print(g.serialize(format='rdf', rdf_format='turtle'))

print('---')

print(g.serialize(format='provn'))