import sqlalchemy
import pandas as pd
import os

# Database Connection String
DB_FOLDER = 'data'
DB_NAME = 'library_data.db'
DB_PATH = os.path.join(os.getcwd(), DB_FOLDER, DB_NAME)
DATABASE_URL = f"sqlite:///{DB_PATH}"

def get_engine():
    """Returns the SQLAlchemy engine."""
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    return sqlalchemy.create_engine(DATABASE_URL)

def save_data_to_db(df, table_name='library_usage', if_exists='replace'):
    """Saves a pandas DataFrame to the SQL database."""
    engine = get_engine()
    df.to_sql(table_name, engine, index=False, if_exists=if_exists)
    print(f"Data saved to table '{table_name}' in {DB_NAME}")

def load_data_from_db(query="SELECT * FROM library_usage"):
    """Loads data from SQL database into a DataFrame."""
    engine = get_engine()
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        return pd.DataFrame() # Return empty if no data yet