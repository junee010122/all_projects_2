import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import yaml
from tqdm import tqdm

def load_config(config_path="configs/params.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_data(dataset_path):
    return pd.read_csv(dataset_path)

def evaluate_preprocessing(X, y, strategy_name):
    """
    Train a simple RandomForest model to evaluate preprocessing choices.
    Returns model accuracy.
    """
    X = X.select_dtypes(include=["number"])  # Ensure only numeric columns are used
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model.score(X, y)

def find_best_n_estimators(X, y):
    """
    Find the optimal number of trees (n_estimators) for Random Forest.
    """
    estimators = [10,50, 100, 200, 500]
    scores = []

    for n in estimators:
        model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        score = cross_val_score(model, X, y, cv=5).mean()  # 5-fold cross-validation
        scores.append(score)

    # Plot accuracy vs. number of trees
    #plt.figure(figsize=(8,5))
    #plt.plot(estimators, scores, marker='o', linestyle='-')
    #plt.xlabel("Number of Trees (n_estimators)")
    #plt.ylabel("Cross-Validation Accuracy")
    #plt.title("Finding Optimal Number of Trees")
    #plt.grid()
    #plt.show()

    return estimators[np.argmax(scores)]  # Return best n_estimators
def preprocess_data(df, config):
    # Separate features and target
    y = df.iloc[:, -1]  # Extract last column as target (Pass/Fail: 1 or -1)
    X = df.iloc[:, :-1]  # Extract all columns except the last one as features

    # Identify numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=["number"]).columns
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns

    # Drop non-numeric columns before anything else
    if len(non_numeric_cols) > 0:
        X.drop(columns=non_numeric_cols, inplace=True)

    best_strategy = None
    best_accuracy = 0

    with tqdm(total=5, desc="Preprocessing Data") as pbar:
        # Handle Missing Values - Compare Strategies
        strategies = ["mean", "median", "drop"]
        for strategy in strategies:
            X_temp = X.copy()
            if strategy != "drop":
                imputer = SimpleImputer(strategy=strategy)
                X_temp[numeric_cols] = imputer.fit_transform(X_temp[numeric_cols])  # Apply only to numeric columns
            else:
                X_temp.dropna(subset=numeric_cols, inplace=True)

            acc = evaluate_preprocessing(X_temp, y, f"Imputation-{strategy}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_strategy = strategy
                X = X_temp.copy()

        print(f"Best Missing Value Strategy: {best_strategy} (Accuracy: {best_accuracy:.4f})")
        pbar.update(1)

        # Feature Selection - Compare Methods
        best_feature_method = None
        feature_methods = ["pca", "mutual_info", "variance_threshold"]
        for method in feature_methods:
            X_temp = X.copy()
            if method == "pca":
                pca = PCA(n_components=10)
                X_temp = pd.DataFrame(pca.fit_transform(X_temp), columns=[f"PC{i}" for i in range(10)])
            elif method == "mutual_info":
                scores = mutual_info_classif(X_temp, y)
                selected_features = X_temp.columns[np.argsort(scores)[-10:]]
                X_temp = X_temp[selected_features]
            elif method == "variance_threshold":
                selector = VarianceThreshold(threshold=0.01)
                X_temp = pd.DataFrame(selector.fit_transform(X_temp))

            acc = evaluate_preprocessing(X_temp, y, f"Feature Selection-{method}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_feature_method = method

        print(f"Best Feature Selection Method: {best_feature_method} (Accuracy: {best_accuracy:.4f})")
        pbar.update(1)

        # Normalize Data - Compare Normalization Techniques
        best_scaler = None
        best_scaler_acc = best_accuracy
        normalization_methods = ["standard", "minmax", "robust"]
        for scaler_type in normalization_methods:
            X_temp = X.copy()
            if scaler_type == "standard":
                scaler = StandardScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            elif scaler_type == "robust":
                scaler = RobustScaler()

            X_temp = pd.DataFrame(scaler.fit_transform(X_temp), columns=X_temp.columns)

            acc = evaluate_preprocessing(X_temp, y, f"Normalization-{scaler_type}")
            if acc > best_scaler_acc:
                best_scaler_acc = acc
                best_scaler = scaler_type
                X = X_temp.copy()

        # Set a default scaler if none improved accuracy
        if best_scaler is None:
            best_scaler = "No Normalization Needed"
        
        print(f"Best Normalization Method: {best_scaler} (Accuracy: {best_scaler_acc:.4f})")
        pbar.update(1)

    # Find the optimal number of trees
    best_n_estimators = find_best_n_estimators(X, y)
    print(f"Optimal Number of Trees: {best_n_estimators}")
    
    # Restore X and y to match original dataframe shape
    processed_df = X.copy()
    processed_df[y.name] = y.values

    # Find the optimal number of trees

    return processed_df  # Return final processed features, target, and original data

