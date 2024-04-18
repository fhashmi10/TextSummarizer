""""Module to perform data transformations"""
import os
from transformers import AutoTokenizer
from datasets import load_from_disk
from src.text_summarizer.entities.config_entity import DataConfig
from src import logger


class DataTransformation:
    """Class to perform data transformation"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def tokenize(self, example_batch):
        """Method to tokenize input batches"""
        try:
            input_encodings = self.tokenizer(
            example_batch['dialogue'], max_length=1024, truncation=True)
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(
                    example_batch['summary'], max_length=128, truncation=True)
            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def transform_data(self):
        """Method to transform data"""
        try:
            if not os.path.exists(self.config.transformed_data_path):
                original_data = load_from_disk(self.config.data_full_path)
                transformed_data = original_data.map(
                    self.tokenize, batched=True)
                transformed_data.save_to_disk(self.config.transformed_data_path)
            else:
                logger.info("Transformed Data already exists at: %s",
                                self.config.transformed_data_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
