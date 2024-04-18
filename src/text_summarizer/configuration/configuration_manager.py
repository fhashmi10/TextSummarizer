"""Module to map data from config to dataclasses"""
from src.text_summarizer.entities.config_entity import (DataConfig)
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
