from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT="-e ."

def get_requirements(file_path:str) -> List[str] :

  """Returns the list of modules in the given file path"""

  requirements = []

  # Reading required modules the file given
  with open(file_path) as file_obj:
    requirements = file_obj.readlines()

    if(HYPEN_E_DOT in requirements) :
      requirements.remove(HYPEN_E_DOT)

  # Removing "\n" from the name of the packages
  requirements = [req.replace("\n","") for req in requirements]

  return requirements

setup(
  name='Diamond Price',
  version='0.0.1',
  author="Himanshu",
  author_email="himanshupandey1036gmail.com",
  install_requires=get_requirements('requirements.txt'),
  packages=find_packages()

)