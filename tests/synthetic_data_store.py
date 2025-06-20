"""
Synthetic Test Datasets for Fuzzy Components

Provides structured test datasets for verifying T-norms, Implicators, and other fuzzy components.
"""

import numpy as np

def get_tnorm_call_testsets():
    return [
        {
            "name": "basic_tnorms_DS_1",
            "a_b": np.array([
                [0.73, 0.18],
                [0.18, 0.73],
                [0.88, 0.88],
                [0.91, 0.48],
                [1.00, 1.00],
                [0.00, 0.00],
                [1.00, 0.65],
                [0.37, 1.00]
            ]),
            "expected": {
                "minimum": np.array([0.18, 0.18, 0.88, 0.48, 1.0, 0.0, 0.65, 0.37]),
                "product": np.array([0.1314, 0.1314, 0.7744, 0.4368, 1.00, 0.00, 0.65, 0.37]),
                "lukasiewicz": np.array([0.0, 0.0, 0.76, 0.39, 1.00, 0.00, 0.65, 0.37]),
                "drastic": np.array([0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.65, 0.37]),
                "hamacher": np.array([0.168764, 0.168764, 0.785714, 0.458246, 1.00, 0.00, 0.65, 0.37]),
                "einstein": np.array([0.107581, 0.107581, 0.763407, 0.417271, 1.00, 0.00, 0.65, 0.37]),
                "nilpotent": np.array([0.00, 0.00, 0.88, 0.48, 1.00, 0.00, 0.65, 0.37]),
                "yager_p=0.835": np.array([0.00, 0.00, 0.724771, 0.332934, 1.00, 0.00, 0.65, 0.37]),
                "yager_p=5.0": np.array([0.179366244, 0.179366244, 0.8621561974, 0.4799838489, 1.00, 0.00, 0.65, 0.37])
            }
        }
    ]

def get_tnorm_reduce_testsets():
    return [
        {
            "name": "tnorm_reduce",
            "similarity_matrix": np.array([
            [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
            [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
            [0.25456, 0.0658,  1.0,     0.3157,  0.53217],
            [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
            [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
        ]),
            "label_mask" : np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0]
        ]),
            "expected": 
            {
                "minimum_outputs": np.array([
                [1.0,     0.2673,  0.0,     0.1197,  0.0],
                [0.2673,  1.0,     0.0,     0.1624,  0.0],
                [0.0,     0.0,     1.0,     0.0,     0.53217],
                [0.1197,  0.1624,  0.0,     1.0,     0.0],
                [0.0,     0.0,     0.53217, 0.0,     1.0]]),
                
                "product_outputs": np.array([
                [1.0,     0.2673,  0.0,     0.1197,  0.0],
                [0.2673,  1.0,     0.0,     0.1624,  0.0],
                [0.0,     0.0,     1.0,     0.0,     0.53217],
                [0.1197,  0.1624,  0.0,     1.0,     0.0],
                [0.0,     0.0,     0.53217, 0.0,     1.0]]),
            
                "luk_outputs" : np.array([
                [1.0,	    0.2673, 0.0,	0.1197,	0.0],
                [0.2673,	1.0,	0.0,	0.1624,	0.0],
                [0.0,	    0.0,	1.0,	0.0,	0.53217],
                [0.1197,	0.1624,	0.0,	1.0,	0.0],
                [0.0,	    0.0,	0.53217,0.0,	1.0]])
            }
        }
    ]

def get_implicator_scalar_testsets():
    return [
        {
            "name": "basic_implicators",
            "a_b": np.array([
                [0.73, 0.18],
                [0.18, 0.73],
                [0.88, 0.88],
                [0.91, 0.48],
                [1.00, 1.00],
                [0.00, 0.00]
            ]),
            "expected": {
                "gaines": np.array([0.24657534246, 1.0, 1.0, 0.527472527, 1.0, 1.0]),
                "goedel": np.array([0.18, 1.0, 1.0, 0.48, 1.0, 1.0]),
                "lukasiewicz": np.array([0.45, 1.00, 1.00, 0.57, 1.00, 1.00]),
                "kleenedienes": np.array([0.27, 0.82, 0.88, 0.48, 1.00, 1.00]),
                "reichenbach": np.array([0.4014, 0.9514, 0.8944, 0.5268, 1.00, 1.00])
            }
        }
    ]


def get_similarity_testing_testsets():
    return [
        {
            "name": "basic_similarity",
            "X": np.array([
            [0.10, 0.32, 0.48],
            [0.20, 0.78, 0.93],
            [0.73, 0.18, 0.28],
            [0.91, 0.48, 0.73],
            [1.00, 0.28, 0.47]
            ]),
            "sigma_for_gaussian_similarity" : 0.67,

            "expected": 
            {
                "sim_matrix_with_linear_similarity_product_tnorm": np.array([
                [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
                [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
                [0.25456, 0.0658,  1.0,       0.3157,  0.53217],
                [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
                [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
            ]),
                "sim_matrix_with_linear_similarity_minimum_tnorm" : np.array([
                [1.00, 0.54, 0.37, 0.19, 0.10],
                [0.54, 1.00, 0.35, 0.29, 0.20],
                [0.37, 0.35, 1.00, 0.55, 0.73],
                [0.19, 0.29, 0.55, 1.00, 0.74],
                [0.10, 0.20, 0.73, 0.74, 1.00]
            ]),
                "sim_matrix_with_linear_similarity_luk_tnorm" : np.array([
                [1.00,	0.00,	0.03,	0.00,	0.05],
                [0.00,	1.00,	0.00,	0.00,	0.00],
                [0.03,	0.00,	1.00,	0.07,	0.44],
                [0.00,	0.00,	0.07,	1.00,	0.45],
                [0.05,	0.00,	0.44,	0.45,	1.00]
            ]),

                "sim_matrix_with_Gaussian_similarity_product_tnorm" : np.array([
                [1.0000,	0.6235,	0.6014,	0.4365,	0.4049],
                [0.6235,	1.0,	0.3059,	0.4935,	0.2932],
                [0.6014,	0.3059,	1.0,	0.6964,	0.8759],
                [0.4365,	0.4935,	0.6964,	1.0,	0.8791],
                [0.4049,	0.2932,	0.8759,	0.8791,	1.0]
            ]),

                "sim_matrix_with_Gaussian_similarity_minimum_tnorm" : np.array([
                [1.,     0.79,   0.6427, 0.4815, 0.4057],
                [0.79,   1.,     0.6246, 0.5704, 0.4902],
                [0.6427, 0.6246, 1.,     0.7981, 0.922 ],
                [0.4815, 0.5704, 0.7981, 1.,     0.9275],
                [0.4057, 0.4902, 0.922,  0.9275, 1.    ]
            ]),

                "sim_matrix_with_Gaussian_similarity_luk_tnorm" : np.array([
                [1.0000,	0.5770,	0.5775,	0.3862,	0.4038],
                [0.5770,	1.0000,	0.0256,	0.4314,	0.0371],
                [0.5775,	0.0256,	1.0000,	0.6673,	0.8715],
                [0.3862,	0.4314,	0.6673,	1.0000,	0.8749],
                [0.4038,	0.0371,	0.8715,	0.8749,	1.0000]
            ])
            }
        }
    ]

def get_ITFRS_testing_testsets():
    Reichenbach_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
    KD_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
    Luk_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
    Goedel_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    Gaines_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    prod_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    min_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])

    return [
        {
            "name": "itfrs",
            "y" : np.array([1, 1, 0, 1, 0]),

            "sim_matrix": np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ]),

            "expected": 
            {
                "Reichenbach_lowerBound" : Reichenbach_lowerBound,
                "KD_lowerBound" : KD_lowerBound,
                "Luk_lowerBound" : Luk_lowerBound,
                "Goedel_lowerBound" : Goedel_lowerBound,
                "Gaines_lowerBound" : Gaines_lowerBound,
                "prod_tn_upperBound" : prod_tn_upperBound,
                "min_tn_upperBound" : min_tn_upperBound

            }
        }
    ]

def get_VQRS_testing_testsets():
    
    return [
        {
            "name": "vqrs",
            "y" : np.array([1, 1, 0, 1, 0]),
            "sim_matrix": np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ]),
            "alpha_lower" : 0.1,
            "beta_lower"  :0.6,
            "alpha_upper" :0.2,
            "beta_upper"  :1.0,

            "expected": 
            {
                "upper_bound_quadratic" : np.array([0.521049496528125,	0.50362976378912,	0.085078125,	0.015835966374605,	0.141019503319767628125]),
                "lower_bound_quadratic" : np.array([1.0,	1.0,	0.5582,	0.2344383779189888,	0.718538097101394872]),
                "upper_bound_linear" : np.array([0.521049496528125,	0.50362976378912,	0.085078125,	0.015835966374605,	0.141019503319767628125]),
                "lower_bound_linear" : np.array([1.0,	1.0,	0.5582,	0.2344383779189888,	0.718538097101394872])

            }
        }
    ]

def get_OWAFRS_testing_testsets():
    pass
    # owa_linear_Reichenbach_lowerBound = np.array([0.822 , 0.8, 0.599, 0.539, 0.624])
    # owa_linear_KD_lowerBound = np.array([0.822 , 0.8, 0.599, 0.539, 0.624])
    # Luk_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
    # Goedel_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # Gaines_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # prod_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    # min_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])

    # return [
    #     {
    #         "name": "owafrs",
    #         "y" : np.array([1, 1, 0, 1, 0]),

    #         "sim_matrix": np.array([
    #         [1.00, 0.54, 0.37, 0.19, 0.10],
    #         [0.54, 1.00, 0.35, 0.29, 0.20],
    #         [0.37, 0.35, 1.00, 0.55, 0.73],
    #         [0.19, 0.29, 0.55, 1.00, 0.74],
    #         [0.10, 0.20, 0.73, 0.74, 1.00]
    #     ]),

    #         "expected": 
    #         {
    #             "Reichenbach_lowerBound" : Reichenbach_lowerBound,
    #             "KD_lowerBound" : KD_lowerBound,
    #             "Luk_lowerBound" : Luk_lowerBound,
    #             "Goedel_lowerBound" : Goedel_lowerBound,
    #             "Gaines_lowerBound" : Gaines_lowerBound,
    #             "prod_tn_upperBound" : prod_tn_upperBound,
    #             "min_tn_upperBound" : min_tn_upperBound
    #         }
    #     }
    # ]