import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            
            # Custom function to replace 'NA' with np.nan
            replace_na_with_nan = lambda X: np.where(X == 'na', np.nan, X)

            # Steps of the Preprocessor Pipeline
            nan_replacement_step = ('nan_replacement', FunctionTransformer(replace_na_with_nan))
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                nan_replacement_step,
                imputer_step,
                scaler_step
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
 
            preprocessor = self.get_data_transformer_object()

            target_column_name = "Good/Bad"
            unwanted_col = "Unnamed: 0"
            target_column_mapping = {1: 0, -1: 1}

            logging.info("Dropping Data")
            #training dataframe
            input_feature_train_df = train_df.drop(columns=[target_column_name,unwanted_col])
            target_feature_train_df = train_df[target_column_name].replace(target_column_mapping)

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[target_column_name,unwanted_col], axis=1)
            target_feature_test_df = test_df[target_column_name].replace(target_column_mapping)

            # Transforming Data using Preprocessor Pipeline
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor.transform(input_feature_test_df)

            logging.info("Data Transformed")
            
            smt = SMOTETomek(sampling_strategy="minority")
            
            logging.info("Value count of target feature is : %s", target_feature_train_df.value_counts())
            
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )
            
            logging.info("Value count of target feature is : %s", target_feature_train_final.value_counts())

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final) ]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df) ]


            save_object(self.data_transformation_config.preprocessor_obj_file_path,
                        obj= preprocessor)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
