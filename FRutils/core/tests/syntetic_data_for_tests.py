import numpy as np

class syntetic_dataset_factory:
    def __init__(self):
        datasets_ = {
        "ds1_X": "ds1",
        "model": "Mustang"
        }

    def implicator_testing_data(self):
        """
        order of columns: a   b
        """
        
        a_b = np.array([
            # [2.10, 4.32],
            # [-0.20, -0.78],
            [0.73, 0.18],
            [0.18, 0.73],
            [0.88, 0.88],
            [0.91, 0.48],
            [1.00, 1.00],
            [0.00, 0.00]
        ])

        gaines_outputs = np.array([0.24657534246, 1.0, 1.0, 0.527472527, 1.0, 1.0])
        goedel_outputs= np.array([0.18, 1.0, 1.0, 0.48, 1.0, 1.0])
        luk_outputs = np.array([0.45, 1.00, 1.00, 0.57, 1.00, 1.00])
        kleene_dienes_outputs = np.array([0.27, 0.82, 0.88, 0.48, 1.00, 1.00])
        reichenbach_outputs = np.array([0.4014, 0.9514, 0.8944, 0.5268, 1.00, 1.00])

        data_dict = {"a_b" : a_b,
                     "gaines_outputs" : gaines_outputs,
                     "goedel_outputs": goedel_outputs,
                     "kleene_dienes_outputs" : kleene_dienes_outputs,
                     "luk_outputs" : luk_outputs,
                     "reichenbach_outputs" : reichenbach_outputs
                     }
        return data_dict
    
    def tnorm_scalar_testing_data(self):
        """
        order of columns: a   b
        """
        
        a_b = np.array([
            # [2.10, 4.32],
            # [-0.20, -0.78],
            [0.73, 0.18],
            [0.18, 0.73],
            [0.88, 0.88],
            [0.91, 0.48],
            [1.00, 1.00],
            [0.00, 0.00]
        ])

        minimum_outputs = np.array([0.18, 0.18, 0.88, 0.48, 1.0, 0.0])
        product_outputs= np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        luk_outputs = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        data_dict = {"a_b" : a_b,
                     "minimum_outputs" : minimum_outputs,
                     "product_outputs": product_outputs,
                     "luk_outputs" : luk_outputs
                     }
        return data_dict
    
    def tnorm_nxnx2_testing_data(self):
        """
        order of maps: nxn[0] mimics similarities therefore square, lower and upper
        triangles are mirrosed. Main diagonal is 1.0
        """
        
        similarity_matrix = np.array([
        
            [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
            [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
            [0.25456, 0.0658,  1.0,       0.3157,  0.53217],
            [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
            [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
        ])

        label_mask = np.array([
            [1., 1., 0., 1., 0.],
            [1., 1., 0., 1., 0.],
            [0., 0., 1., 0., 1.],
            [1., 1., 0., 1., 0.],
            [0., 0., 1., 0., 1.]
        ])

        nxnx2_map = np.stack([similarity_matrix, label_mask], axis=-1)

        minimum_outputs = np.array([
            [1.0,     0.2673,  0.0,     0.1197,  0.0],
            [0.2673,  1.0,     0.0,     0.1624,  0.0],
            [0.0,     0.0,     1.0,     0.0,     0.53217],
            [0.1197,  0.1624,  0.0,     1.0,     0.0],
            [0.0,     0.0,     0.53217, 0.0,     1.0]
        ])
        product_outputs= np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        luk_outputs = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        data_dict = {"nxnx2_map" : nxnx2_map,
                     "minimum_outputs" : minimum_outputs,
                     "product_outputs": product_outputs,
                     "luk_outputs" : luk_outputs
                     }
        return data_dict

    def get_ds1(self):
        ds1_X = np.array([
            [0.10, 0.32, 0.48],
            [0.20, 0.78, 0.93],
            [0.73, 0.18, 0.28],
            [0.91, 0.48, 0.73],
            [1.00, 0.28, 0.47]
        ])

        ds1_y = np.array([1, 1, 0, 1, 0])

        label_mask_calc = np.array([
            [1., 1., 0., 1., 0.],
            [1., 1., 0., 1., 0.],
            [0., 0., 1., 0., 1.],
            [1., 1., 0., 1., 0.],
            [0., 0., 1., 0., 1.]
        ])
        sim_matrix_calc = np.array([
            [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
            [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
            [0.25456, 0.0658,  1.0,       0.3157,  0.53217],
            [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
            [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
        ])


        
        return ds1_X, ds1_y