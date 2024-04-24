"""Module to define data classes for config mapping"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Class to map data config"""
    source_url: str
    download_path: Path
    data_path: Path
    data_full_path: Path
    tokenizer_name: str
    transformed_data_path: Path


@dataclass(frozen=True)
class ModelConfig:
    """Class to map model config"""
    model_checkpoint_name: str
    model_checkpoint_path: Path
    trained_model_path: Path


@dataclass(frozen=True)
class ParamConfig:
    """Class to map param config"""
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    gradient_accumulation_steps: int


@dataclass
class EvaluationConfig:
    """Class to map evaluation config"""
    eval_metrics: str
    eval_metrics_type: list
    eval_scores_path: Path
