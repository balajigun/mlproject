import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("========== Training Pipeline Started ==========")

            # Data Ingestion
            logging.info("Starting Data Ingestion...")
            ingestion = DataIngestion()

            train_path, test_path = ingestion.initiate_data_ingestion()

            logging.info("Data Ingestion Completed")

            # Data Transformation
            logging.info("Starting Data Transformation...")

            transformation = DataTransformation()

            train_arr, test_arr, preprocessor_path = (
                transformation.initiate_data_transformation(
                    train_path,
                    test_path
                )
            )

            logging.info("Data Transformation Completed")

            # Model Training
            logging.info("Starting Model Training...")

            trainer = ModelTrainer()

            r2_score = trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info("Model Training Completed")

            logging.info(f"Best Model R2 Score : {r2_score:.4f}")

            logging.info("========== Pipeline Finished Successfully ==========")

        except Exception as e:
            logging.exception("Error occurred while executing training pipeline.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()