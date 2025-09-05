# src/components/model_trainer.py

import os
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def initiate_model_trainer(self, train_array, test_array):
        # training logic here
        # save the trained model at self.config.trained_model_file_path
        return "Training complete"
