from imblearn.over_sampling.base import BaseOverSampler
from abc import ABC, abstractmethod
from FRsutils.utils.init_helpers import assign_allowed_kwargs
from FRsutils.utils.validation_utils import _get_fr_model_param_schema

from FRsutils.utils.validation_utils import (
    validate_choice,
    validate_fr_model_params,
    ALLOWED_FR_MODELS,
    ALLOWED_SIMILARITIES,
    ALLOWED_TNORMS,
    ALLOWED_RANKING_STRATEGIES
)

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
                 similarity_tnorm_name='minimum',
                 instance_ranking_strategy_name='pos',
                 sampling_strategy='auto',
                 **kwargs):
        
        super().__init__(sampling_strategy=sampling_strategy)
        
        validate_fr_model_params(fr_model_name, kwargs)

        fr_model_schema = _get_fr_model_param_schema(fr_model_name)
        assign_allowed_kwargs(self, kwargs, fr_model_schema)
        

        
        # Validate string options and assign to class attributes
        fr_model_name = validate_choice("fr_model_name", fr_model_name, ALLOWED_FR_MODELS)
        self.fr_model_name = fr_model_name
        
        similarity_name = validate_choice("similarity_name", similarity_name, ALLOWED_SIMILARITIES)
        self.similarity_name = similarity_name
        
        similarity_tnorm_name = validate_choice("similarity_tnorm", similarity_tnorm_name, ALLOWED_TNORMS)
        self.similarity_tnorm_name = similarity_tnorm_name
        
        
        instance_ranking_strategy_name = validate_choice("instance_ranking_strategy", instance_ranking_strategy_name, ALLOWED_RANKING_STRATEGIES)
        self.instance_ranking_strategy_name = instance_ranking_strategy_name


    
    @abstractmethod 
    def _build_internal_objects(self, X, y):
        """
        @brief creates the internal objects of the resampler.
        all classes should implement this method
        
        @return None
        """
        pass
    
    @abstractmethod 
    def fit_resample(self, X, y):
        """
        @brief Fits the resampler to the data and returns the resampled data.
        all classes should implement this method
        
        @param X The input data.
        @param y The target labels.
        @return A tuple of the resampled data and labels.
        """
        pass
 