#Here we willn transform the categorical data into numerical data.
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')#So here inside artifact preprocessor.pickel file would be created.

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()#This variable will contain preprocessor_obj_file_path.
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.

        '''
        try:
            numerical_columns=['writing_score','reading_score']#here there would be all numerical features
            categorical_columns=[#There would be all categorical features
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"

            
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )#This is a numerical pipeline in which null values would be replaced by median and also be performing StandardScaler on numerical_pipeline.

            #lets create a categorical pipeline

            categorical_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy='most_frequent')),
                       ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')),
                       ('scaler',StandardScaler(with_mean=False))]
            )#there are 3 transformation applied on categorical columns 
            # first is SimpleImputer which will substitute null values with most_frequent values i.e mode
            #Then we apply oneHotEncoder
            #Then we apply StandardScaler.

            logging.info(f"categorical columns :{categorical_columns}")
            logging.info(f"Numerical columns:{numerical_columns}")
            #Lets combine both the pipelines using columntransformer
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),#numerical pipeline on numerical columns
                    ("cat_columns",categorical_pipeline,categorical_columns)#categorical pipeline on categorical columns.
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)#reading training data
            test_df=pd.read_csv(test_path)#reading test data

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()#this function which we created will be used for transforming training and test data.

            target_column_name='math_score' #this is the o/p data
            numerical_column=["writing_score","reading_score"]

            #lets create dependent and independent on training data
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)#here all input data would be there
            target_feature_train_df=test_df[target_column_name]#here all the o/p data would be present
            #lets create dependent and independent on testing data
            input_feature_test_df=train_df.drop(columns=[target_column_name],axis=1)#here all input data would be there
            target_feature_test_df=test_df[target_column_name]#here all the o/p data would be present




            logging.info(f"Applying preprocessing object on training dataframe and testing datafcrame")
            #transforming training and test data.

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)#Applying transformation on train data
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)#Applying transformation on test data.

            train_arr=np.c_[input_feature_train_arr , np.array(target_feature_train_df)]#this will contain training input data and training target data
            test_arr=np.c_[input_feature_test_arr , np.array(target_feature_test_df)]#This will contain testing input data and testing o/p data
            #hstack is used to combine matrix with different dimensions.
             
            #train_arr and test_arr would be in the form of numpy array.

            logging.info(f"saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )#So we want to store all of this transformation in a pickel file.
            #hence we are going to save the pickel file.
            return(
                train_arr,
                test_arr,

                self.data_transformation_config
            )
        except Exception as e:
            raise CustomException(e,sys)



