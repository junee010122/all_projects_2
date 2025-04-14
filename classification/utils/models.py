import os
import numpy as np
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_model(params=None):
    return LogisticRegression(**params) if params else LogisticRegression(max_iter=1000)

def objective(trial, X, y, params):
    if params['model'] == 0:
        model = LogisticRegression(
            C=trial.suggest_float('C', *params['optimization']['logistic_regression']['C']),
            max_iter=1000,
            class_weight='balanced'
        )
    elif params['model'] == 1:
        model = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', *params['optimization']['xgboost']['n_estimators']),
            max_depth=trial.suggest_int('max_depth', *params['optimization']['xgboost']['max_depth']),
            learning_rate=trial.suggest_float('learning_rate', *params['optimization']['xgboost']['learning_rate']),
            scale_pos_weight=trial.suggest_float('scale_pos_weight', *params['optimization']['xgboost']['scale_pos_weight'])
        )

    model.fit(X, y)
    y_pred = model.predict(X)
    return recall_score(y, y_pred)  # Maximize recall

def train_model(X, y, params):
    result_path = params['paths']['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    def wrapped_objective(trial):
        return objective(trial, X_train, y_train, params)

    study = optuna.create_study(direction='maximize')
    study.optimize(wrapped_objective, n_trials=100, show_progress_bar=True)

    print("Best trial:")
    print(f"  Recall Score: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Final model with best parameters
    if params['model'] == 0:
        model = LogisticRegression(
            C=study.best_params['C'],
            max_iter=1000,
            class_weight='balanced'
        )
        model_type = 'logistic_regression'
    elif params['model'] == 1:
        model = XGBClassifier(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            learning_rate=study.best_params['learning_rate'],
            scale_pos_weight=study.best_params['scale_pos_weight']
        )
        model_type = 'xgboost'

    model.fit(X_train, y_train)

    os.makedirs(result_path, exist_ok=True)
    model_filename = os.path.join(result_path, f"{model_type}_model.pkl")
    joblib.dump(model, model_filename)

    return model

