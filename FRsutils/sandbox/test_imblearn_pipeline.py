import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline

from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE  # <-- update path
from sklearn.model_selection import GridSearchCV

# # -----------------------------
# # Synthetic imbalanced dataset
# # -----------------------------
# X, y = make_classification(
#     n_samples=500,
#     n_features=10,
#     n_informative=5,
#     n_redundant=2,
#     weights=[0.9, 0.1],
#     random_state=42
# )

# # -----------------------------
# # FRSMOTE config
# # -----------------------------
# fr_model_params = {
#     'lb_tnorm': 'minimum',
#     'ub_implicator': 'lukasiewicz',
# }

# frsmote = FRSMOTE(
#     fr_model_name='ITFRS',
#     similarity_name='linear',
#     similarity_tnorm_name='minimum',
#     instance_ranking_strategy_name='pos',
#     sampling_strategy='auto',
#     k_neighbors=5,
#     bias_interpolation=True,
#     random_state=42,
#     **fr_model_params
# )

# # -----------------------------
# # Create imblearn pipeline
# # -----------------------------
# pipeline = Pipeline([
#     ('scaler', MinMaxScaler()),   # Normalize to [0, 1]
#     ('frsmote', frsmote),         # Your custom oversampler
#     ('svc', SVC(probability=True))
# ])

# # -----------------------------
# # Evaluate with cross-validation
# # -----------------------------
# scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
# print("F1 scores (CV):", scores)
# print("Mean F1:", np.mean(scores))

#############################################################

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.9, 0.1],
    random_state=42)

aa = MinMaxScaler()
X_train = aa.fit_transform(X)

# Example fuzzy-rough params
fr_model_params = {
    'lb_tnorm': 'minimum',
    'ub_implicator': 'goedel',
    'X': X_train.copy()  # or inject this dynamically before resampling
}

# Define pipeline
pipeline = Pipeline([
    ('sampling', FRSMOTE(fr_model_name='ITFRS', fr_model_params=fr_model_params)),
    ('classifier', RandomForestClassifier())
])

# Define param grid
param_grid = {
    'sampling__k_neighbors': [3, 5],
    'sampling__fr_model_name': ['ITFRS'],
    'classifier__n_estimators': [50, 100]
}

# Run GridSearchCV
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro')
grid.fit(X_train, y)
