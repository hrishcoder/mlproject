from flask import Flask,request,render_template

import numpy as np
import pandas as pd
from sklear.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)
@app.route("/")#this means home page.
def index():
    return render_template('index.html')#so all the html code that has been written in index.html will be displayed inside home page.

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template('home.html')#if method==GET it means no data has been stored in the server.If thats the case go to the home page and fill that data and then the method would be considered as post.

    else:#if method == POST then
         data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )#reques.form.get() will extract the input data from the server and then give it to CustomData class.
         
         pred_df=data.get_data_as_data_frame()#After that this method will apply transformation function on dataframe and then apply model prediction on that dataframe.
         print(pred_df)
         print("Before Prediction")

         predict_pipeline=PredictPipeline()#After converting the data to dataframe then we give that data for prediction purpose.So this function is used for data transformation and data model.
         print("Mid Prediction")
         results=predict_pipeline.predict(pred_df)#this will give dataframe to that function and then transform the data as well as 
         print("after Prediction")
         return render_template('home.html',results=results[0])#Then give this result to  home.html  and all the result would be displayed on home.html.
    if __name__=="__main__":
        app.run(host="0.0.0.0")       
        

 