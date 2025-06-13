from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Step 1: Create imbalanced data
X, y = make_classification(n_samples=300, n_features=10, n_informative=6,
                           n_redundant=2, weights=[0.85, 0.15], random_state=42)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = np.clip(X, 0.0, 0.99)

# Step 2: Define pipeline with custom FRSMOTE and SVC
pipe = Pipeline([
    ("frsmote", FRSMOTE(
        type="itfrs",  # fuzzy rough model type
        k_neighbors=5,
        bias_interpolation=False,
        random_state=42
    )),
    ("svc", SVC())
])

# Step 3: Define hyperparameter grid
param_grid = {
    # FRSMOTE + ITFRS parameters
    "frsmote__similarity": ["gaussian", "linear"],
    "frsmote__similarity_tnorm": ["minimum", "product"],
    "frsmote__gaussian_similarity_sigma": [0.1, .4],
    "frsmote__tnorm_name": ["minimum"],
    "frsmote__implicator_name": ["lukasiewicz"],

    # Optional: SVC parameters
    "svc__C": [0.1],
    "svc__kernel": ["rbf"]
}

# Step 4: Run grid search
grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=2)
grid.fit(X, y)

# Extract all results as a DataFrame
results_df = pd.DataFrame(grid.cv_results_)
results_df.to_excel("gridsearch_results.xlsx", index=False)

# Step 5: Results
print("Best Params:\n", grid.best_params_)
print("Best F1 Score:", grid.best_score_)
