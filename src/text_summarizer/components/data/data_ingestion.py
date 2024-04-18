""""Module to perform data ingestion"""
import os
from urllib import request
import zipfile
from src.text_summarizer.entities.config_entity import DataConfig
from src.utils.common import create_directories
from src import logger


class DataIngestion:
    """Class to perform data ingestion"""

    def __init__(self, config: DataConfig):
        self.config = config

    def extract_zip_file(self):
        """Extracts the zip file into the data directory"""
        try:
            create_directories([self.config.data_path])
            with zipfile.ZipFile(self.config.download_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.data_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def ingest_data(self):
        """Method to ingest data"""
        try:
            if self.config.source_url is not None:
                if not os.path.exists(self.config.download_path):
                    create_directories([self.config.download_path], is_file_path=True)
                    filename, headers = request.urlretrieve(
                        url=self.config.source_url,
                        filename=self.config.download_path
                    )
                    self.extract_zip_file()
                    logger.info("Data downloaded at: %s",
                                self.config.download_path)
                else:
                    logger.info("Data will be read from: %s",
                                self.config.download_path)
            else:
                logger.info("Data will be read from: %s",
                            self.config.download_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
