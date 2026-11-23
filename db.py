import os, pandas as pd
from sqlalchemy import create_engine
E_=create_engine(os.getenv("DATABASE_URL","sqlite:///quant.db"))
def save(t,row): pd.DataFrame([row]).to_sql(t,E_,if_exists="append",index=False)
def load(t):
    try: return pd.read_sql(f"select * from {t}",E_)
    except Exception: return pd.DataFrame()
