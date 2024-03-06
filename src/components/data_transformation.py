import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

from sklearn.impute import SimpleImputer # Handling Missing Values
from sklearn.preprocessing import StandardScaler # Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding

## pipelines
from sklearn.pipeline import Pipeline # Making Pipeline
from sklearn.compose import ColumnTransformer # Mergeing Two Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

## Data Transformation Config
@dataclass
class DataTransormationConfig:
  preporcessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl') # file path of pipeline pickel

## Data Transformation class
class DataTransformation:
  def __init__(self):
    self.data_transformation_config=DataTransormationConfig()

  def get_data_transformation_object(self):
    """Making Pipeline of the preprocessing"""
    try :
      logging.info('Data Transormation intitiated')

      # Segrigation numerical and categorical variables
      categorical_cols = ['cut', 'color','clarity']
      numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

      # Define the custom ranking for each ordinal variable
      cut_categories = ["Fair","Good","Very Good","Ideal","Premium"]
      clarity_categories = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
      color_categories = ["D","E","F","G","H","I","J"]

      ## Numerical Pipeline
      num_pipleline=Pipeline(
          steps=[
              ('imputer',SimpleImputer(strategy='median')),
              ('scaler',StandardScaler())
          ]
      )

      ## Categorical Pipleline
      cat_pipleline=Pipeline(
          steps=[
              ('imputer',SimpleImputer(strategy='most_frequent')),
              ('ordinalencoder',OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
              ('scaler',StandardScaler())
          ]
      )

      preprocessor=ColumnTransformer([
          ('num_pipeline',num_pipleline,numerical_cols),
          ('cat_pipeline',cat_pipleline,categorical_cols)
      ])

      logging.info('Pipeline Created')
      return preprocessor

    except Exception as e:
      logging.error('Error Occured in creating pipeline')
      raise CustomException(e,sys)
    
  
  def initate_data_transormation(self, train_path, test_path) :
    try:
      # Reading train and test data
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)

      logging.info('Read train and test data completed')
      logging.info(f"Train Dataframe Head : \n {train_df.head(3).to_string()}")
      logging.info(f"Test Dataframe Head : \n {test_df.head(3).to_string()}")

      logging.info('Obtaining preprocessing object')

      preprocessing_obj = self.get_data_transformation_object()
    
      target_column_name = 'price'
      drop_columns = [target_column_name,'id']

      # feature into independent and dependent features
      input_feature_train_df = train_df.drop(columns=drop_columns)
      target_feature_train_df = train_df[target_column_name]

      input_feature_test_df = test_df.drop(columns=drop_columns)
      target_feature_test_df = test_df[target_column_name]

      # apply the transformation
      input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

      train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
      test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

      save_object(
        file_path=self.data_transformation_config.preporcessor_ob_file_path,
        obj=preprocessing_obj
      )

      logging.info('Precoessor pickle is created and saved')

      return (
        train_arr,
        test_arr,
        self.data_transformation_config.preporcessor_ob_file_path
      )

    except Exception as e:
      logging.error('Error occured while tranforming train and test data using pipeline')
      raise CustomException(e,sys)