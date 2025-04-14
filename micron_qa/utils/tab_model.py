import numpy as np
import os
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier



def get_model(model_type):
    if model_type == 'logistic_regression':
        return LogisticRegression(max_iter=1000)
    elif model_type == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    else:  # default to xgboost
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )


def train_model(data, params):
    X, y = data['features'], data['labels']

    model_map = {0: 'xgboost', 1: 'logistic_regression', 2: 'mlp'}
    model_type_code = params['model']['tabular'].get('model_type', 0)
    model_type = model_map.get(model_type_code, 'xgboost')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = get_model(model_type)
    model.fit(X_train, y_train)


    # Save model
    result_path = params['paths'].get('results', './results')
    
    if params['training'].get('use_optuna', 0) == 1:
        return train_with_optuna(X_train, X_test, y_train, y_test, model_type, result_path)

    os.makedirs(result_path, exist_ok=True)
    model_filename = os.path.join(result_path, f'{model_type}_model.pkl')
    joblib.dump(model, model_filename)

    return model


def evaluate_model(model, data, params):
    X, y = data['features'], data['labels']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

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

