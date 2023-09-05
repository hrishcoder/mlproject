#This is step-1.
from setuptools import find_packages,setup
from typing import List

HYPHEN_DOT='-e .'

def get_requirements(file_path:str)->List[str]:#This means file_path will take requirement.txt file as a value and thwe data inside of it would be in  the form of string and the function will return list which would have string elements.
    requirements=[]#requirement list will contain list of all packages
    with open(file_path) as file_obj:#open the file requirement.txt
        requirements=file_obj.readlines()#read the content line by line
        requirements=[req.replace("\n","") for req in requirements]#Since it will contain \n replace it with '' and there would be list od packages

        if HYPHEN_DOT in requirements:
            requirements.remove(HYPHEN_DOT)#Since -e . is used to incvoke setup after that we want it to be removed from the list.
    
    return requirements #so it will return list of packages.


setup(

    name='mlproject',#name of the project
    version='0.0.1',#version that is being used
    author='Hrishab',#Author of the project
    author_email='hrishabshetty@gmail.com',#email id
    packages=find_packages(),#packages that are going to be used or finds how many packages are there,So here src is a package since it contains __init__.py
    install_requires=get_requirements('requirements.txt')#installation that is required.Since there could be multiple packages that will be installed so we will create a function which takes requirement.txt file has a parameter
)