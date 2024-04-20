"""Module to train models"""
import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, TrainingArguments, Trainer
from src.text_summarizer.entities.config_entity import DataConfig, ModelConfig, ParamConfig
from src import logger


class ModelTrainer:
    """Class to train models"""

    def __init__(self, data_config: DataConfig,
                 model_config: ModelConfig, param_config: ParamConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.param_config = param_config

    @staticmethod
    def get_trained_tokenizer(tokenizer_name: str):
        """Method to re-train, save and return tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            return tokenizer
            # re-train tokenizer
            # https://huggingface.co/learn/nlp-course/en/chapter6/2
            # def get_training_corpus():
            #     return (
            #         raw_datasets["train"][i : i + 1000]["whole_func_string"]
            #         for i in range(0, len(raw_datasets["train"]), 1000)
            #     )
            # training_corpus = get_training_corpus()
            # old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
            # tokenizer.save_pretrained("code-search-net-tokenizer")
            # tokenizer = AutoTokenizer.from_pretrained("path/code-search-net-tokenizer")
            # use this one then
        except AttributeError as ex:
            logger.exception("Error processing tokenizer.")
            raise ex
        except Exception as ex:
            raise ex

    @staticmethod
    def get_model(model_checkpoint: str):
        """Method to get the model"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_checkpoint).to(device)
            return pretrained_model
        except AttributeError as ex:
            logger.exception("Error loading model.")
            raise ex
        except Exception as ex:
            raise ex

    def train_model(self):
        """Method to invoke model training"""
        try:
            trained_tokenizer = self.get_trained_tokenizer(
                self.data_config.tokenizer_name)
            pretrained_model = self.get_model(
                self.model_config.model_checkpoint_name)

            seq2seq_data_collator = DataCollatorForSeq2Seq(trained_tokenizer,
                                                           model=pretrained_model)

            # Load data
            data = load_from_disk(self.data_config.data_full_path)

            trainer_args = TrainingArguments(
                output_dir=self.model_config.model_checkpoint_path,
                num_train_epochs=self.param_config.num_train_epochs,
                warmup_steps=self.param_config.warmup_steps,
                per_device_train_batch_size=self.param_config.per_device_train_batch_size,
                per_device_eval_batch_size=self.param_config.per_device_train_batch_size,
                weight_decay=self.param_config.weight_decay,
                logging_steps=self.param_config.logging_steps,
                evaluation_strategy=self.param_config.evaluation_strategy,
                eval_steps=self.param_config.eval_steps,
                save_steps=self.param_config.save_steps,
                gradient_accumulation_steps=self.param_config.gradient_accumulation_steps
            )

            model_trainer=Trainer(model=pretrained_model,
                              args=trainer_args,
                              tokenizer=trained_tokenizer,
                              data_collator=seq2seq_data_collator,
                              train_dataset=data["train"],
                              eval_dataset=data["validation"])
            model_trainer.train()

            # Save tokenizer
            trained_tokenizer.save_pretrained(self.model_config.trained_tokenizer_path)
            # Save model
            model_trainer.save_model(self.model_config.trained_model_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
