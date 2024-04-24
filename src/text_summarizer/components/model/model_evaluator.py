from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm
from src.text_summarizer.entities.config_entity import \
    DataConfig, ModelConfig, ParamConfig, EvaluationConfig
from src import logger


class ModelEvaluator:
    """Class to evaluate model"""

    def __init__(self, data_config: DataConfig, model_config: ModelConfig,
                 params: ParamConfig, eval_config: EvaluationConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.params = params
        self.eval_config = eval_config

    def get_trained_model(self, device):
        """Method to get the trained model"""
        try:
            trained_tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.trained_model_path)
            trained_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_config.trained_model_path).to(device)
            return trained_tokenizer, trained_model
        except AttributeError as ex:
            logger.exception("Error loading trained model.")
            raise ex
        except Exception as ex:
            raise ex

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        try:
            for i in range(0, len(list_of_elements), batch_size):
                yield list_of_elements[i: i + batch_size]
        except Exception as ex:
            raise ex

    def evaluate_model(self):
        """Method to invoke model evaluation"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Get trained model
            trained_tokenizer, trained_model = self.get_trained_model(
                device=device)

            # Get test data
            transformed_data = load_from_disk(self.data_config.transformed_data_path)
            test_data = transformed_data["test"][0:10]
            # Just first 10 rows for quick testing of evaluation

            # Eval Metrics
            rouge_metric = load_metric(self.eval_config.eval_metrics)

            # Evaluate
            article_batches = list(self.generate_batch_sized_chunks(
                test_data['dialogue'], batch_size=2))
            target_batches = list(self.generate_batch_sized_chunks(
                test_data['summary'], batch_size=2))

            for article_batch, target_batch in tqdm(
                    zip(article_batches, target_batches), total=len(article_batches)):
                inputs = trained_tokenizer(article_batch, max_length=1024,  truncation=True,
                                           padding="max_length", return_tensors="pt")
                summaries = trained_model.generate(input_ids=inputs["input_ids"].to(device),
                                                   attention_mask=inputs["attention_mask"].to(
                                                       device),
                                                   length_penalty=0.8, num_beams=8, max_length=128)
                # length_penalty penalized long summarization

                # Finally, we decode the generated texts,
                # replace the  token, and add the decoded texts with the references to the metric.
                decoded_summaries = [trained_tokenizer.decode(s, skip_special_tokens=True,
                                                              clean_up_tokenization_spaces=True)
                                     for s in summaries]
                decoded_summaries = [d.replace("", " ")
                                     for d in decoded_summaries]

                rouge_metric.add_batch(predictions=decoded_summaries,
                                 references=target_batch)

            #  Finally compute and return the ROUGE scores.
            score = rouge_metric.compute()

            rouge_dict = dict((rn, score[rn].mid.fmeasure)
                              for rn in self.eval_config.eval_metrics_type)

            df = pd.DataFrame(rouge_dict, index=['pegasus'])
            df.to_csv(self.eval_config.eval_scores_path, index=False)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
