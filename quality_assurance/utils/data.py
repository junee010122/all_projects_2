import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

def create_database():
    engine = create_engine("mysql+mysqlconnector://user:password@localhost/secom")
    return engine

def load_data_to_sql(csv_path, table_name):
    engine = create_database()
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"Data loaded into table {table_name}")

def preprocess_data(sql_engine):
    query = """
    SELECT * FROM secom_table;
    """
    df = pd.read_sql(query, sql_engine)
    df.fillna(df.mean(), inplace=True)  # Example: Handling missing values
    return df