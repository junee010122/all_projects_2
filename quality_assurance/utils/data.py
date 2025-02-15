import pandas as pd
import mysql.connector
from tqdm import tqdm
from sqlalchemy import create_engine
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import yaml

def evaluate_preprocessing(X, y, strategy_name):
    X = X.select_dtypes(include=["number"])  # Ensure only numeric columns are used
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model.score(X, y)

def handle_missing_values(strategies, X, y, numeric_cols):

    best_strategy = None
    best_accuracy = 0

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
    
    return X, y

def handle_feature_selection(strategies, X, y):

    best_accuracy = 0 
    best_feature_method = None

    for method in strategies:
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
    return X, y

def handle_normalization(strategies, X, y):

    best_scaler = None
    best_scaler_acc = 0
    for scaler_type in strategies:
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

    if best_scaler is None:
        best_scaler = "No Normalization Needed"
        
    print(f"Best Normalization Method: {best_scaler} (Accuracy: {best_scaler_acc:.4f})")
    
    return X, y

def handle_imbalance():
    pass

def create_SQL(data):

    # This code is just an example to get and idea of
    # converting CSV files into sql database
    # The created database itself won't be necessarily
    # used for training a model, or running ML techniques

    USER = "root"  
    PASSWORD = "junyoung4u"  
    HOST = "localhost"
    PORT = "3306"
    DATABASE = "secom"  
    TABLE_NAME = "secom_data"

    conn = mysql.connector.connect(
            host = HOST,
            user = USER,
            password = PASSWORD
            )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS secom;")
    conn.commit()
    conn.close()
    engine = create_engine(f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")

    data.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
    print("successfully converted to SQL")


def preprocess_data(data, params):

    # read in params
    impute_strategy = params['preprocessing']['missing_value']
    feature_selection_strategy = params['preprocessing']['feature_selection']
    normalization_strategy = params['preprocessing']['normalization']
    imbalance_strategy = params['preprocessing']['imbalance']

    
    create_SQL(data)
    df = data
    y = df.iloc[:, -1]  # Extract last column as target (Pass/Fail: 1 or -1)
    X = df.iloc[:, :-1]  # Extract all columns except the last one as features
    
    numeric_cols = X.select_dtypes(include=["number"]).columns
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns
     
    if len(non_numeric_cols) > 0:
        X.drop(columns=non_numeric_cols, inplace=True)
    

    with tqdm(total=5, desc='Preprocessing Data') as pbar:
        
        X,y = handle_missing_values(impute_strategy, X, y, numeric_cols)
        X,y = handle_feature_selection(feature_selection_strategy, X, y)
        X,y = handle_normalization(normalization_strategy, X, y)
        X,y = handle_imbalance(imbalance_strategy, X, y)
   
    processed_df = X.copy()
    processed_df[y.name] = y.values()

    return processed_df
