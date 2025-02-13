import pandas as pd
import mysql.connector
import tqdm
from sqlalchemy import create_engine

def evaluate_preprocessing(X, y, strategy_name):
    X = X.select_dtypes(include=["number"])  # Ensure only numeric columns are used
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model.score(X, y)

def handle_missing_values(X, y, numeric_cols):
    pass

def handle_feature_selection(X, y):
    pass

def handle_normalization(X, y):
    pass

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
    
    best_strategy = None
    best_accuracy = 0

    with tqdm(total=5, desc='Preprocessing Data') as pbar:
        
        handle_missing_values(impute_strategy, X, y, numeric_cols)
        handle_feature_selection(feature_selection_strategy, X, y)
        handle_normalization(normalization_strategy, X, y)
        handle_imbalance(imbalance_strategy, X, y)
    
