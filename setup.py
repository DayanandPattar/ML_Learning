from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT="-e ."
def get_requirements(file_path:str)->List[str]:
    ''' 
    this function will return list of requirements from file provided
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements
    

setup(
name="MLTutorialProject",
version="0.0.1",
author="Dayanand Pattar",
author_email="daya6252@gmail.com",
find_packages=find_packages(),
install_requires=['pandas','numpy','seaborn']
)