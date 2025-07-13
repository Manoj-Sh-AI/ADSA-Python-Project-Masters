from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str)->list[str]:
    """
    This function will return the list of requirements

    Args:
        file_path (str): _description_

    Returns:
        list[str]: _description_
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", " ") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name="ADSA_Python_project",
    version="0.0.1",
    author="Manoj Saligrama Harisha",
    author_email="shmanoj2002@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)