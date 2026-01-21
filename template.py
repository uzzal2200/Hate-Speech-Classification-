import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


list_of_files = [
    "flowcharts",
    "hate/components/__init__.py",
    "hate/components/data_ingestion.py",
    "hate/components/data_transformation.py",
    "hate/components/model_evaluation.py",
    "hate/components/model_pusher.py",
    "hate/components/model_trainer.py",
    "hate/configuration/__init__.py",
    "hate/configuration/gcloud_syncer.py",
    "hate/constants/__init__.py",
    "hate/entity/__init__.py",
    "hate/entity/artifact_entity.py",
    "hate/entity/config_entity.py",
    "hate/exception/__init__.py",
    "hate/logger/__init__.py",
    "hate/pipeline/__init__.py",
    "hate/pipeline/model.py",
    "hate/pipeline/ml/__init__.py",
    "hate/pipeline/ml/model.py",
    "notebook/",
    ".gitignore",
    "Dockerfile",
    "app.py",
    "requirements.txt",
    "setup.py",
    "demo.py",
   
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")