import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

##import custom logger and exception handeling class
from exception import CustomException
from logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        logging.info("Saving obj as pickle")
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Saving obj as pickle completed")
    except Exception as e:
        raise CustomException(e, sys)