"""Define config file paths"""
from pathlib import Path
from src import logger
from src.text_summarizer.configuration.configuration_manager import ConfigurationManager

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")

def get_config():
    """Method to load trained model"""
    try:
        config = ConfigurationManager(config_file_path=CONFIG_FILE_PATH,
                                      params_file_path=PARAMS_FILE_PATH)
        return config
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
