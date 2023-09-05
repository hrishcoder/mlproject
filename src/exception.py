#So this was step-2.
import sys
from src.logger import logging
#Placeholder:it is represented as [{}] and it is commonly used in string functions.
#So what it does is replace [{}] with something using format function.
#eg:format(3,hello)
#So [{}] will get replaced with 3 and hello
#[{}] hi [{}]
#after using format it would be 3 hi hello.

def error_message_detail(error,error_detail:sys):
    _,_,exec_tb=error_detail.exc_info()#so this function will give us the information about the error,eg:Which line the error is,which file the error is and so on.
    file_name=exec_tb.tb_frame.f_code.co_filename#it will give us the name of the file.
    error_message="Error occured in python script name [{0}] line number[{1}] error message[{2}]".format(
        file_name,exec_tb.tb_lineno,str(error)


    )
    return error_message





#exec_tb.tb_lineno->This will give us the line number where the error has occured.
'''
The format parameter will have 3 things 
1.filename
2.line number
3.error message
file name is represented as exec_tb.tb_frame.f_code.co_filename
lineno is represented as exec_tb.tb_lineno
error message is repreasented as str(error)
so it will five us the information on which file the error is i.e filename
on which line the error is i.e lineno
and what is the error str(error)
'''

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):#we created a constructor which contains 2 parameters error_message and error_detail.
        super().__init__(error_message)#using super.__init__ function we will give error_message to the exception class,which will do the later computation
        self.error_message=error_message_detail(error_message,error_detail=error_detail)#This function will have 2 parameters error_message and error_detail.
    
    def __str__(self):#when we raise the custom exception it will raise the error message.
        return self.error_message

'''
So whenever we raise the exception a warning message would be shownb that is built by us,the warning 
message is

syntax: error_message="Error occured in python script name [{0}] line number[{1}] error message[{2}]".format(
        file_name,exec_tb.tb_lineno,str(error)


    )

'''
if __name__ =="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide By Zero")
        raise CustomException(e,sys)
    
'''So logger and exceptions both are working perfectly'''