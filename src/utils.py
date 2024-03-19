import os
import sys
import pymongo

import numpy as np
import pandas as pd

from pymongo import MongoClient

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

from dotenv import load_dotenv
load_dotenv("env/.env")


def import_collection_as_dataframe(collection_name, db_name):
    try:
        logging.info("Entered Import Collection")

        mongo_client = MongoClient(os.getenv("MONGO_CONNECTION_URL"))
        collection = mongo_client[db_name][collection_name]

        df = pd.DataFrame(list(collection.find()))
        
        logging.info("First few element of df: %s", df.head())
        logging.info("Data converted to dataframe")

        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        return df

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            data = dill.load(file_obj)
            return data

    except Exception as e:
        raise CustomException(e, sys)


def download_model(dest_file_name):
    try:

        return dest_file_name

    except Exception as e:
        raise CustomException(e, sys)





def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
