import numpy as np

class syntetic_dataset_factory:
    def __init__(self):
        datasets_ = {
        "ds1_X": "ds1",
        "model": "Mustang"
        }

    def get_ds1(self):
        ds1_X = np.array([
            [0.10, 0.32, 0.48],
            [0.20, 0.78, 0.93],
            [0.73, 0.18, 0.28],
            [0.91, 0.48, 0.73],
            [1.00, 0.28, 0.47]
        ])
        ds1_y = np.array([1, 1, 0, 1, 0])
        
        return ds1_X, ds1_y