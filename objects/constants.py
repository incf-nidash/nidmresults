"""
Definition of constants.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

from prov.model import Namespace
from prov.model import PROV

NIDM = Namespace('nidm', "http://www.incf.org/ns/nidash/nidm#")
NIIRI = Namespace("niiri", "http://iri.nidash.org/")
CRYPTO = Namespace("crypto", "http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions#")
FSL = Namespace("fsl", "http://www.incf.org/ns/nidash/fsl#")
DCT = Namespace("dct", "http://purl.org/dc/terms/")

GAUSSIAN_DISTRIBUTION = NIDM['GaussianDistribution']

INDEPEDENT_CORR = NIDM['IndependentError']
SERIALLY_CORR = NIDM['SeriallyCorrelatedError']
COMPOUND_SYMMETRY_CORR = NIDM['CompoundSymmetricError']
ARBITRARILY_CORR = NIDM['ArbitriralyCorrelatedError']

CORRELATION_ENUM = {
    INDEPEDENT_CORR,
    SERIALLY_CORR,
    COMPOUND_SYMMETRY_CORR,
    ARBITRARILY_CORR
}

SPATIALLY_GLOBAL = NIDM['SpatiallyGlocal']
SPATIALLY_LOCAL = NIDM['SpatiallyLocal']
SPATIALLY_REGUL = NIDM['SpatiallyRegularized']

SPATIAL_DEPENDENCY_ENUM = {
    SPATIALLY_GLOBAL, 
    SPATIALLY_LOCAL,
    SPATIALLY_REGUL
}

ESTIMATION_OLS = NIDM['OrdinaryLeastSquares']
ESTIMATION_WLS = NIDM['WeightedLeastSquares']
ESTIMATION_GLS = NIDM['GeneralizedLeastSquares']