import os
import json
import pandas as pd
from pymongo.mongo_client import MongoClient

# loading environment in our Python script
from dotenv import load_dotenv
load_dotenv("env/.env")

uri = os.getenv("MONGO_CONNECTION_URL")
client = MongoClient(uri)

DATABASE_NAME=os.getenv("MONGO_DATABASE_NAME")
COLLECTION_NAME=os.getenv("MONGO_COLLECTION_NAME")

df=pd.read_csv(r"notebooks/wafer_Data.csv")

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)