import numpy as np

class syntetic_dataset_factory:
    def __init__(self):
        pass

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
        product_outputs= np.array([0.1314, 0.1314, 0.7744, 0.4368, 1.00, 0.00])
        luk_outputs = np.array([0.00, 0.00, 0.76, 0.39, 1.00, 0.00])

        data_dict = {"a_b" : a_b,
                     "minimum_outputs" : minimum_outputs,
                     "product_outputs": product_outputs
                     ,"luk_outputs" : luk_outputs
                     }
        return data_dict
    
    def tnorm_nxnx2_testing_dataset(self):
        """
        order of maps: nxn[0] mimics similarities therefore square, lower and upper
        triangles are mirrosed. Main diagonal is 1.0
        This is with binary masks matrix. Therefore tnorm min and product will act the same. SO we need another dataset
        """
        
        similarity_matrix = np.array([
        
            [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
            [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
            [0.25456, 0.0658,  1.0,       0.3157,  0.53217],
            [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
            [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
        ])

        label_mask = np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0]
        ])

        minimum_outputs = np.array([
            [1.0,     0.2673,  0.0,     0.1197,  0.0],
            [0.2673,  1.0,     0.0,     0.1624,  0.0],
            [0.0,     0.0,     1.0,     0.0,     0.53217],
            [0.1197,  0.1624,  0.0,     1.0,     0.0],
            [0.0,     0.0,     0.53217, 0.0,     1.0]
        ])
        product_outputs = np.array([
            [1.0,     0.2673,  0.0,     0.1197,  0.0],
            [0.2673,  1.0,     0.0,     0.1624,  0.0],
            [0.0,     0.0,     1.0,     0.0,     0.53217],
            [0.1197,  0.1624,  0.0,     1.0,     0.0],
            [0.0,     0.0,     0.53217, 0.0,     1.0]
        ])

        luk_outputs = np.array([
            [1.0,	    0.2673, 0.0,	0.1197,	0.0],
            [0.2673,	1.0,	0.0,	0.1624,	0.0],
            [0.0,	    0.0,	1.0,	0.0,	0.53217],
            [0.1197,	0.1624,	0.0,	1.0,	0.0],
            [0.0,	    0.0,	0.53217,0.0,	1.0]
        ])

        data_dict = {"similarity_matrix" : similarity_matrix,
                     "label_mask" : label_mask,
                     "minimum_outputs" : minimum_outputs,
                     "product_outputs": product_outputs
                     ,"luk_outputs" : luk_outputs
                     }
        return data_dict
    

    
    

 
    


    def owa_weights_linear_testing_data(self):
        """
        owa weights
        """
        owa_infimum_weights_linear_len_5 = np.array([0.06666667, 0.13333333, 0.2, 0.26666667, 0.33333333])
        owa_infimum_weights_linear_len_10 = np.array([0.01818182, 0.03636364, 0.05454545, 0.07272727, 0.09090909, 0.10909091, 0.12727273, 0.14545455, 0.16363636, 0.18181818])
        
        owa_suprimum_weights_linear_len_8 = np.array([0.22222222, 0.19444444, 0.16666667, 0.13888889, 0.11111111, 0.08333333, 0.05555556, 0.02777778])
        owa_supriimum_weights_linear_len_13 = np.array([0.14285714, 0.13186813, 0.12087912, 0.10989011, 0.0989011,  0.08791209, 0.07692308, 0.06593407, 0.05494505, 0.04395604, 0.03296703, 0.02197802, 0.01098901])
        

        data_dict = {"owa_infimum_weights_linear_len_5" : owa_infimum_weights_linear_len_5,
                     "owa_infimum_weights_linear_len_10" : owa_infimum_weights_linear_len_10,
                     "owa_suprimum_weights_linear_len_8":owa_suprimum_weights_linear_len_8,
                     "owa_suprimum_weights_linear_len_13":owa_supriimum_weights_linear_len_13}
        return data_dict