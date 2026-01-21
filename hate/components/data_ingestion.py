import os
import sys
import shutil
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """Load the local dataset into the artifacts directory."""
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of DataIngestion class")
        try:
            source_path = self.data_ingestion_config.SOURCE_DATA_PATH
            destination_dir = self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR
            os.makedirs(destination_dir, exist_ok=True)

            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Dataset not found at {source_path}")

            destination_path = self.data_ingestion_config.INGESTED_FILE_PATH
            shutil.copyfile(source_path, destination_path)
            logging.info(f"Copied dataset to {destination_path}")

            data_ingestion_artifacts = DataIngestionArtifacts(raw_data_path=destination_path)
            logging.info("Exited the initiate_data_ingestion method of DataIngestion class")
            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e