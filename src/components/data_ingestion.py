import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

## intialize the data ingestion configuration -> responsible for intializing params for the data ingestion class
@dataclass
class DataIngestionConfig:
  artifact="artifacts"
  train_data_path=os.path.join("artifacts","train.csv")
  test_data_path=os.path.join("artifacts","test.csv")
  raw_data_path=os.path.join("artifacts","raw.csv") # if reading data from data base then temporarialy store data

## create  a data ingestion class -> responsible for reading and train test split
class DataIngestion:
  def __init__(self):
    self.ingestion_config=DataIngestionConfig()

  def initiate_data_ingestion(self):
    logging.info("Data Ingestion Method starts")

    try:
      logging.info(os.path.join('data','gemstone.csv'))
      df=pd.read_csv(os.path.join('data','gemstone.csv'))
      logging.info("Dataset read as pandas DataFrame")
      
      os.makedirs(self.ingestion_config.artifact, exist_ok=False)
      df.to_csv(self.ingestion_config.raw_data_path, index=False)

      logging.info("Train Test Split")
      train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

      train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
      test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

      logging.info("Ingestion of data is completed")

      return (
        self.ingestion_config.train_data_path,
        self.ingestion_config.test_data_path
      )

    except Exception as e:
      logging.info("Error occured in Data Ingestion Config")
      raise CustomException(e,sys)