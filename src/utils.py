import numpy as np
import pandas as pd
import os
import sys

from src.exception import CustomException
import dill as pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)#path of the directory

        os.makedirs(dir_path,exist_ok=True)#This will make the directory.

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
