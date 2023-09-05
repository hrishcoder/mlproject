'''
Logging:Any exxecution that happens on log or store those information on some files so that we could track if there are
any errors,any exception that occurs we will store that in a text file.
'''

#So this is w.r.t logging setup
#need to learn more about this.
import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    force=True,
    level=logging.INFO,
)

if __name__ =="__main__":
   logging.info("Logging has started")


