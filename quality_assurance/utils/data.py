import pandas as pd
import mysql.connector
from sqlalchemy import create_engine



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


