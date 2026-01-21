import os
import sys
import shutil
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts, ModelTrainerArtifacts


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_trainer_artifacts: ModelTrainerArtifacts):
        """Copy the trained model to a stable saved_model directory for inference."""
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifacts = model_trainer_artifacts

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logging.info("Entered initiate_model_pusher")
        try:
            os.makedirs(self.model_pusher_config.SAVED_MODEL_DIR, exist_ok=True)
            shutil.copyfile(
                self.model_trainer_artifacts.trained_model_path,
                self.model_pusher_config.SAVED_MODEL_PATH,
            )
            tokenizer_dest = os.path.join(
                self.model_pusher_config.SAVED_MODEL_DIR, os.path.basename(self.model_trainer_artifacts.tokenizer_path)
            )
            shutil.copyfile(self.model_trainer_artifacts.tokenizer_path, tokenizer_dest)

            logging.info(
                f"Saved model and tokenizer copied to {self.model_pusher_config.SAVED_MODEL_DIR}"
            )

            model_pusher_artifact = ModelPusherArtifacts(
                saved_model_path=self.model_pusher_config.SAVED_MODEL_PATH
            )
            logging.info("Exited the initiate_model_pusher method")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e