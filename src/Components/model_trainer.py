import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")#So here we are creating a pickeln file for training the model.

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()#So this variable will contain  artifacts file path.

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],#input training data
                train_array[:,-1],#output training data
                test_array[:,:-1],#input test data
                test_array[:,-1]#output test data
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                 }#we are going to try each and every model and select the best model based on its performance metrics
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)#So we will create this function inside utils.py and we will pass 5 parameters
            #X_train,X_test,y_train,y_test and the model these are the parameters we are giving to the model.
            #The model in return will return the performance metrics of all model.

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]#this variable will get the best_model from best_model_score and after getting the best_model_name through best_model_score, we will get that best_model from the dictionary.
            best_model = models[best_model_name]#Get the best model from the models dictionary

            if best_model_score<0.6:#if the best_model_score is less the 0.6 then raise an exception.
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            #now we are going to save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)#checking if the model works perfectly on test data
            r2_square=r2_score(y_test,predicted)#CVhecking if the models performance metric using r2_score.
            return  r2_square #this will return performance metric of the model.

        except Exception as e:
            raise CustomException(e,sys)


       
    

