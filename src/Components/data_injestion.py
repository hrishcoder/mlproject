#it will contain code related reading the data from the database.

#So data injestion means extracting the data from either csv file or other databases like hadoop,sql,from there you can extract data.,

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')#datanInjestionConfig will take train_data_path as input and as an o/p would be stored in the artifact folder.
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    #so it will take train_data_path,test_data_path and raw_data_path as an input and as an o/p it would be stored in the artifact.

class DataInjestion:
    def __init__(self) :
        self.injestion_config=DataIngestionConfig()#This variable will contain all three paths,i.e train_data_path,test_data_path and raw_data_path

    def initiate_data_ingestion(self):#in this method we will read data from csv or other database.
        logging.info("Entered the data injestion method or component.")
        try:
            df=pd.read_csv('notebook\data\stud.csv')#So read the data from the csv file.
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)#make directory from train_data_path
            df.to_csv(self.injestion_config.raw_data_path,index=False,header=True)#the csv data would be stored in the raw_data_path.
            logging.info('Train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)#training test split would occur on entire dataset
            train_set.to_csv(self.injestion_config.train_data_path,index=False,header=True)#Train data would be stored inside train_data_path
            test_set.to_csv(self.injestion_config.test_data_path,index=False,header=True)#Test data path would be stored inside test_data_path.
            #csv would be stored inside raw_data
            #train_data would be stored inside train_data_path
            #test_data would be stored inside test_data_path.
            logging.info("Ingestion of the data is completed.")

            return(
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )#train_data_path will contain train data and test_data_path will contain test data.
        #So this train_data_path and test_data_path will be used for data transformation.


        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataInjestion()
    obj.initiate_data_ingestion()

