import os
import sys
import pandas as pd

# user defined
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
  obj=DataIngestion()
  train_data_path, test_data_path = obj.initiate_data_ingestion()
  print(train_data_path,test_data_path)

  data_transformation=DataTransformation()
  train_arr,test_arr,_=data_transformation.initate_data_transormation(train_path=train_data_path,test_path=test_data_path)
  
  model_trainer = ModelTrainer()
  model_trainer.initiate_model_traning(train_arr=train_arr,test_arr=test_arr)
