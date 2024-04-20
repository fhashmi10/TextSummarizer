"""Module to create training pipeline"""
import sys
from src import logger
from src.text_summarizer.configuration import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.text_summarizer.configuration.configuration_manager import ConfigurationManager
from src.text_summarizer.components.data.data_ingestion import DataIngestion
from src.text_summarizer.components.data.data_transformation import DataTransformation
from src.text_summarizer.components.model.model_trainer import ModelTrainer
from src.text_summarizer.components.model.model_evaluator import ModelEvaluator


class TextSummarizerTrainingPipeline:
    """Class to create training pipeline"""

    def __init__(self):
        pass

    def data_ingestion(self, config: ConfigurationManager, stage_name: str):
        """Method to perform data ingestion"""
        try:
            logger.info("%s started", stage_name)
            data_ingestion = DataIngestion(config=config.get_data_config())
            data_ingestion.ingest_data()
            logger.info("%s completed\nx==========x", stage_name)
        except Exception as ex:
            raise ex

    def data_transformation(self, config: ConfigurationManager, stage_name: str):
        """Method to perform data transformation"""
        try:
            logger.info("%s started", stage_name)
            data_transformation = DataTransformation(
                config=config.get_data_config())
            data_transformation.transform_data()
            logger.info("%s completed\nx==========x", stage_name)
        except Exception as ex:
            raise ex

    def model_trainer(self, config: ConfigurationManager, stage_name: str):
        """Method to perform model training"""
        try:
            logger.info("%s started", stage_name)
            model_trainer = ModelTrainer(data_config=config.get_data_config(),
                                         model_config=config.get_model_config(),
                                         params=config.get_param_config())
            model_trainer.train_model()
            logger.info("%s completed\nx==========x", stage_name)
        except Exception as ex:
            raise ex

    def model_evaluator(self, config: ConfigurationManager, stage_name: str):
        """Method to perform model evaluation"""
        try:
            logger.info("%s started", stage_name)
            # model_evaluator = ModelEvaluator(data_config=config.get_data_config(),
            #                                  model_config=config.get_model_config(),
            #                                  params=config.get_param_config(),
            #                                  eval_config=config.get_evaluation_config())
            # model_evaluator.evaluate_model()
            logger.info("%s completed\nx==========x", stage_name)
        except Exception as ex:
            raise ex

    def run_pipeline(self, steps: int, stage: str = ""):
        """Method to perform training
           Stage is needed to distinguish multiple projects under src if any"""
        try:
            if steps != 0:
                config = ConfigurationManager(config_file_path=CONFIG_FILE_PATH,
                                              params_file_path=PARAMS_FILE_PATH)
                if steps >= 1:
                    self.data_ingestion(
                        config=config, stage_name=stage+": Data Ingestion")
                if steps >= 2:
                    self.data_transformation(
                        config=config, stage_name=stage+": Data Transformation")
                if steps >= 3:
                    self.model_trainer(
                        config=config, stage_name=stage+": Model Training")
                # if steps >= 4:
                #     self.model_evaluator(
                #         config=config, stage_name=stage+": Model Evaluation")
            else:
                logger.info(
                    "Please provide number of steps to run min 1 to max 4.")
        except Exception as ex:
            raise ex


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            NUM_STEPS = int(sys.argv[1])
        else:
            NUM_STEPS = 0
        obj = TextSummarizerTrainingPipeline()
        obj.run_pipeline(steps=NUM_STEPS)
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
