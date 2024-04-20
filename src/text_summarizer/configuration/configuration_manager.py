"""Module to map data from config to dataclasses"""
from src.text_summarizer.entities.config_entity import DataConfig, ModelConfig, ParamConfig
from src.utils.common import read_yaml_configbox
from src import logger


class ConfigurationManager:
    """Class to manage configuration"""

    def __init__(self, config_file_path, params_file_path):
        try:
            self.config = read_yaml_configbox(config_file_path)
            self.params = read_yaml_configbox(params_file_path)
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def get_data_config(self) -> DataConfig:
        """Method to manage data configuration"""
        try:
            config = self.config.data
            data_config = DataConfig(source_url=config.source_url,
                                     download_path=config.download_path,
                                     data_path=config.data_path,
                                     data_full_path=config.data_full_path,
                                     tokenizer_name=config.tokenizer_name,
                                     transformed_data_path=config.transformed_data_path)
            return data_config
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def get_model_config(self) -> ModelConfig:
        """Method to manage model configuration"""
        try:
            config = self.config.model
            model_config = ModelConfig(model_checkpoint_name=config.model_checkpoint_name,
                                       model_checkpoint_path=config.model_checkpoint_path,
                                       trained_model_path=config.trained_model_path)
            return model_config
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def get_param_config(self) -> ParamConfig:
        """Method to manage param configuration"""
        try:
            params = self.params
            param_config = ParamConfig(eval_steps=params.eval_steps,
                                       evaluation_strategy=params.evaluation_strategy,
                                       gradient_accumulation_steps=\
                                        params.gradient_accumulation_steps,
                                       logging_steps=params.logging_steps,
                                       num_train_epochs=params.num_train_epochs,
                                       per_device_train_batch_size=\
                                        params.per_device_train_batch_size,
                                       save_steps=params.save_steps,
                                       warmup_steps=params.warmup_steps,
                                       weight_decay=params.weight_decay)
            return param_config
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
