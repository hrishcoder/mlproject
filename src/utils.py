import numpy as np
import pandas as pd
import os
import sys

from src.exception import CustomException
import dill 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)#path of the directory

        os.makedirs(dir_path,exist_ok=True)#This will make the directory.

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):#converting dictionary of models into list of models and then iterating through each model.
            model = list(models.values())[i]#So this is where the model would be initialized i.e all the models in the dictionary
            #Hyperparameter tuning
            para=param[list(models.keys())[i]]
            #Apply grid search cv

            gs = GridSearchCV(model,para,cv=3)
            #Use this grid search cv on X_train and y_train.
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)#it will contain all the best hyperparameter values for that model. 
            model.fit(X_train,y_train)

           

            
            #model.fit(X_train,y_train)#Then we fit the model on training data

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)#Here we will check the prediction of training data for accuracy purpose

            y_test_pred = model.predict(X_test)#Here we will check the prediction of testing data.

            train_model_score = r2_score(y_train, y_train_pred)#Performance metrics of training model

            test_model_score = r2_score(y_test, y_test_pred)#Performance metrics of testing model

            report[list(models.keys())[i]] = test_model_score

            return report#this will return performance metric of each and every model.
    except:
        pass
