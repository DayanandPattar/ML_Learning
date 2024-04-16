import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# for @dataclass decorator
from dataclasses import dataclass

##import custom logger and exception handeling class
from exception import CustomException
from logger import logging

## import from data transformation
from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig

## import model trainer
from components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts", "train.csv"
    )  # path to save train data
    test_data_path: str = os.path.join(
        "artifacts", "test.csv"
    )  # path to save test data
    raw_data_path: str = os.path.join(
        "artifacts", "raw_data.csv"
    )  # path to save raw input data


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Component")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read data csv as dataframe")
            
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            logging.info("Saving input data as csv for reference")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test Split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Saving Train data as csv")
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            logging.info("Saving Test data as csv")
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Train Test Split Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


# code to test data ingestion block
if __name__ == "__main__":
    logging.info("Testing Data Ingestion")
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
