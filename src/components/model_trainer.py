import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object


# Model Trainer config -> path of model.pkl file
@dataclass
class ModelTrainerConfig:
  model_ob_file_path=os.path.join('artifacts','model.pkl')

# Model Trainer
class ModelTrainer:
  def __init__(self):
    self.model_traning_config=ModelTrainerConfig()
  
  def initiate_model_traning(self,train_arr, test_arr):
    try:
      logging.info('Model Training started')
      X_train,y_train,X_test,y_test = (
        train_arr[:,:-1],
        train_arr[:,-1],
        test_arr[:,:-1],
        test_arr[:,-1],
      )

      models = {
        'LinearRegression':LinearRegression(),
        'Lasso':Lasso(),
        'Ridge' : Ridge(),
        'DecisionTreeRegressor' : DecisionTreeRegressor(),
        'RandomForestRegressor':RandomForestRegressor(),
        'AdaBoostRegressor' : AdaBoostRegressor(),
        'ElasticNet':ElasticNet()
      }

      model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)

      print(model_report)
      print('\n')
      print("="*50)
      logging.info(f"Model Report :\n{model_report}")

      best_model_score = max(sorted(model_report.values()))

      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]

      best_model = models[best_model_name]

      print(f'\nBest Model is : {best_model_name}, R2 Score :{best_model_score}')
      print('='*50)
      logging.info(f'\nBest Model is : {best_model_name}, R2 Score :{best_model_score}')

      save_object(
        file_path=self.model_traning_config.model_ob_file_path,
        obj=best_model
      )

    except Exception as e:
      logging.error('Error occured while Traning the model')
      raise CustomException(e,sys)