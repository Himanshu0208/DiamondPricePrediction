import os
import sys
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
      dir_path = os.path.dirname(file_path)

      os.makedirs(dir_path, exist_ok=True)

      with open(file_path,'wb') as file_obj:
        pickle.dump(obj=obj,file=file_obj)

    except Exception as e:
      logging.error('Error Occured while saving the file')
      raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
  try:
    report={}
    for i in range(len(list(models))):
      model=list(models.values())[i]
      model.fit(X_train,y_train)

      #Make Predictions
      y_pred=model.predict(X_test)

      # mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
      # rmse = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_pred))
      r2_square = r2_score(y_true=y_test,y_pred=y_pred)
      report[list(models.keys())[i]] = r2_square

    return report

      # r2_list.append({"model":model, "R2 Score":r2_square})
  except Exception as e:
    logging.error('Error Occured while evaluation model')
    raise CustomException(e,sys)