from imblearn.over_sampling.base import BaseOverSampler
from abc import ABC, abstractmethod

class BaseAllPurposeFuzzyRoughOversampler(ABC, BaseOverSampler):
    """
    @brief Abstract base class for oversampling using Fuzzy Rough Sets.

    @details This base class is intended to be inherited by resamplers that either:
     - Use fuzzy-rough set theory directly for ranking or selecting instances.
     - Combine fuzzy-rough logic with generative models (e.g., VAE, GAN).
     - Use fuzzy-rough sets as a preprocessing step for other resamplers.
     - The class is designed to be flexible and can be extended to support different
    types of fuzzy-rough sets and oversampling strategies.

    It should not be used directly.

    @warning Do not instantiate or use this class directly. Use one of its concrete subclasses instead.

    @inherits BaseOverSampler
    @inherits ABC
    """
    def __init__(self,
                 fr_model_name='ITFRS',
                 similarity_name='linear',
                 similarity_tnorm='lukasiewicz',
                 instance_ranking_strategy='pos',
                 sampling_strategy='auto',
                 **kwargs):
        
        super().__init__(sampling_strategy=sampling_strategy)
        
        self.fr_model_name = fr_model_name
        self.similarity_name = similarity_name
        self.similarity_tnorm = similarity_tnorm
        self.instance_ranking_strategy = instance_ranking_strategy
        self.fr_model_params = kwargs['fr_model_params']
      
    @abstractmethod 
    def supported_strategies(self):
        """
        @brief Returns a list of supported sampling strategies.
        all classes should implement this method
        
        @return A list of supported sampling strategies.s
        """
        pass    