## Create a generic template for any project used in the industry
import os
from pathlib import Path
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(lineno)d %(message)s]"

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
)

project_name="fire_smoke_detection"

list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py", # convert src folder into a package
    f"src/{project_name}/components/__init__.py", # components are the building blocks of the project (data ingestion, data validation, data transformation, model training, model evaluation, model deployment, model monitoring, model retraining, model rollback, model versioning, model registry, model serving)
    f"src/{project_name}/utils/__init__.py", # utils are the helper functions used in the project
    f"src/{project_name}/utils/common.py", # common.py is the main utility file which are common to all functionalities
    f"src/{project_name}/config/__init__.py", # config are the configuration files used in the project
    f"src/{project_name}/config/config.yaml", # config.yaml is the main configuration file
    f"src/{project_name}/config/configuration.py", # configuration.py is the main configuration file
    f"src/{project_name}/pipeline/__init__.py", # pipeline is the main pipeline of the project
    f"src/{project_name}/entity/__init__.py", # entity is the main entity file
    f"src/{project_name}/entity/config_entity.py", # config_entity.py is the main configuration entity file
    f"src/{project_name}/constants/__init__.py", # constants are the constants used in the project
    "config/config.yaml", # all the configuration details
    "params.yaml", # all the parameters details for machine learning training
    "schema.yaml", # all the schema details 
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "templates/index.html",
]


for filepath in list_of_files:
    filepath=Path(filepath)
    filedir=filepath.parent
    filedir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Creating the directory {filedir} and the file {filepath}")
    if not filepath.exists():
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating the file {filepath}")
    else:
        logging.info(f"File {filepath} already exists")