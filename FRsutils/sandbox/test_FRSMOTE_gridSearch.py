from sklearn.model_selection import GridSearchCV

from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE  # <-- update path


param_grid = {
    "lb_tnorm": ['product', 'minimum'],
    "ub_implicator": ['lukasiewicz'],
    "k_neighbors": [3, 5, 7]
}

grid = GridSearchCV(FRSMOTE(...), param_grid, cv=3)
grid.fit(X, y)