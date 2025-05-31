from FRsutils.core.tnorms import MinTNorm, ProductTNorm, LukasiewiczTNorm
from FRsutils.core.similarities import LinearSimilarity, GaussianSimilarity
import FRsutils.core.implicators as impl

from FRsutils.core.config_tnorm import TNormConfig
from FRsutils.core.config_similarity import SimilarityConfig
from FRsutils.core.config_implicator import ImplicatorConfig


def build_tnorm(config: TNormConfig):
    return {
        'min': MinTNorm(),
        'product': ProductTNorm(),
        'lukasiewicz': LukasiewiczTNorm()
    }[config.type]

def build_similarity(config: SimilarityConfig):
    if config.type == 'linear':
        return LinearSimilarity()
    elif config.type == 'gaussian':
        return GaussianSimilarity(sigma=config.sigma)

def build_implicator(config: ImplicatorConfig):
    return {
        'gaines': impl.imp_gaines,
        'goedel': impl.imp_goedel,
        'kleene': impl.imp_kleene_dienes,
        'reichenbach': impl.imp_reichenbach,
        'lukasiewicz': impl.imp_lukasiewicz
    }[config.type]
