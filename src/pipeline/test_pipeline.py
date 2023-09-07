import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predictPipeline:
    def __init__(self):
        pass

class CustomData:#This class is responsible for mapping the input data to the backend
     def __init__( self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):#This is the information about the i/p feilds

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
     def predict(self,features):#This function is for predicting the model
         try:
            model_path=os.path.join("artifacts","model.pkl")#this pkl file is responsible for predicting the model 
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')#This pkl file is responsible for transforming the data
            print("Before Loading")
            #load: it means get that file and use it
            model=load_object(file_path=model_path)#load the model pkl file i.e get the model pkl file and use it for prediction
            preprocessor=load_object(file_path=preprocessor_path)#load the preprocessor pkl file i.e get the preprocessor pkl file and use it for transforming the data
            print("After Loading")
            data_scaled=preprocessor.transform(features)#Transform the data
            preds=model.predict(data_scaled)#do model prediction on the data
            return preds
         except Exception as e:
           raise CustomException(e,sys)

     def get_data_as_data_frame(self):#This function will convert the data that we got from the i/p to dataframe
        try:
            custom_data_input_dict = {#Store that data in the form of dictionary
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)#Then convert the data into dataframe.

        except Exception as e:
            raise CustomException(e, sys)