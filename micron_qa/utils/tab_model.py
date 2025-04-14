import numpy as np
import os
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_model(model_type, params=None):
    if model_type == 'logistic_regression':
        return LogisticRegression(**params) if params else LogisticRegression(max_iter=1000)
    else:  # default to xgboost
        return XGBClassifier(**params) if params else XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )


def train_model(data, params):
    X, y = data['features'], data['labels']

    model_map = {0: 'xgboost', 1: 'logistic_regression'}
    model_type_code = params['model']['tabular'].get('model_type', 0)
    model_type = model_map.get(model_type_code, 'xgboost')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    result_path = params['paths'].get('results', './results')
    use_optuna = params['training'].get('use_optuna', 0)
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    if use_optuna:
        def objective(trial):

            if model_type == 'xgboost':
                rng = params['model']['optuna']['xgboost']
                trial_params = {
                    'n_estimators': trial.suggest_int('n_estimators', *rng['n_estimators']),
                    'max_depth': trial.suggest_int('max_depth', *rng['max_depth']),
                    'learning_rate': trial.suggest_float('learning_rate', *rng['learning_rate']),
                    'subsample': trial.suggest_float('subsample', *rng['subsample']),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', *rng['colsample_bytree']),
                    'scale_pos_weight': num_neg / num_pos,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
            else:
                rng = params['model']['optuna']['logistic_regression']
                trial_params = {
                    'C': trial.suggest_float('C', *rng['C']),
                    'max_iter': 1000,
                    'class_weight': 'balanced'
                }

            model = get_model(model_type, trial_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return recall_score(y_test, y_pred, zero_division=0)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        print("Best trial:")
        print(f"  F1 Score: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")
       
        best_params = study.best_params
        if model_type == 'xgboost':
            best_params.update({
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'scale_pos_weight': 5
        })
        elif model_type == 'logistic_regression':
            best_params['class_weight'] = 'balanced'

        model = get_model(model_type, best_params)
        model.fit(X_train, y_train)

        os.makedirs(result_path, exist_ok=True)
        model_filename = os.path.join(result_path, f"{model_type}_optuna_model.pkl")
        joblib.dump(model, model_filename)
        return model

    else:
        model = get_model(model_type)
        model.fit(X_train, y_train)
        os.makedirs(result_path, exist_ok=True)
        model_filename = os.path.join(result_path, f"{model_type}_model.pkl")
        joblib.dump(model, model_filename)
        return model


def evaluate_model(model, data, params):
    X, y = data['features'], data['labels']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.2  # << LOWER THRESHOLD TO CATCH MORE FAILS
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("[Evaluation Metrics]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

