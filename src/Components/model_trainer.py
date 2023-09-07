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
            
            #we are creating dictionary of hyperparameters for hyperparameter tuning purpose
            params={
                "Decision Tree": {#hyperparameters for decision tree
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{#hyperparameters for Random Forest
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{#hyperparameters for Gradient Boosting
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                 "Linear Regression":{},
                "XGBRegressor":{#hyperparameters for XGBRegressor
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{#hyperparameters for CatBoosting Regressor
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{#hyperparameters for AdaBoost Regressor
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            #give all these hyperparameters to evaluate_models function for hyperparameter tuning
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)#So we will create this function inside utils.py and we will pass 5 parameters
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


       
    

