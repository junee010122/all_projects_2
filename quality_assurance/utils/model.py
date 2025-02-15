import pandas as pd
import numpy as np
import yaml

from tqdm import tqdm
from scipy.stats import f_oneway
from sklearn.feature_selection import f_classif

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def run_stats(x, y):
    pass

def feature_selection(X: pd.DataFrame, y: pd.Series, params):

    print("Running ANOVA for Feature Selection with JMP...")

    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y must be a pandas Series"
    F_values, p_values = f_classif(X, y)

    # Store p-values in a dictionary
    p_values_dict = {X.columns[i]: p_values[i] for i in range(len(p_values))}

    # Sort features by importance (lower p-values = more significant)
    sorted_features = sorted(p_values_dict.items(), key=lambda x: x[1])

    # Step 1️⃣ Get Accuracy of Full Model
    best_model, baseline_accuracy = train_model(X, y, params)

    # Step 2️⃣ Iteratively Select Features Until Accuracy is Maintained
    best_features = []
    for feature, p_value in sorted_features:
        best_features.append(feature)

        # Train model with selected features
        X_reduced = X[best_features]
        y_reduced = y.loc[X_reduced.index]  # Ensure y matches X

        _, accuracy = train_model(X_reduced, y_reduced, params)

        # Stop selecting if accuracy is within 1% of the baseline
        if abs(baseline_accuracy - accuracy) / baseline_accuracy < 0.01:
            break

    print(f"Selected {len(best_features)} features to maintain accuracy close to original.")

    # Return selected features and corresponding target
    X_selected = X[best_features]
    y_selected = y.loc[X_selected.index]

    return X_selected, y_selected


def train_model(x, y, params):

    model_types = params['task']['models']
    best_model = None
    best_score = -np.inf
    best_model_name = ""

    if set(np.unique(y)) == {-1, 1}:
        y = (y + 1) // 2 
    # Define models based on params.yaml
    models = {
        "logistic_regression": LogisticRegression(C=params["task"]["hyperparameters"]["logistic_regression"]["C"]),
        "random_forest": RandomForestClassifier(
            n_estimators=params["task"]["hyperparameters"]["random_forest"]["n_estimators"],
            max_depth=params["task"]["hyperparameters"]["random_forest"]["max_depth"]
        ),
        "xgboost": XGBClassifier(
            n_estimators=params["task"]["hyperparameters"]["xgboost"]["n_estimators"],
            learning_rate=params["task"]["hyperparameters"]["xgboost"]["learning_rate"],
            max_depth=params["task"]["hyperparameters"]["xgboost"]["max_depth"]
        )
    }

    print("Evaluating Models...")
    
    for model_name in model_types:
        model = models[model_name]

        scores = cross_val_score(model, x, y, cv=5, scoring="accuracy")

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = model_name

    print(f"Best Model: {best_model_name} (Accuracy: {best_score:.4f})")
    best_model.fit(x, y)

    return best_model, best_model_name
