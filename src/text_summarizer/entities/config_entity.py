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
    model_url: str
    base_model_path: Path
    built_model_path: Path
    transform_model_path: Path
    trained_model_path: Path


@dataclass(frozen=True)
class ParamConfig:
    """Class to map param config"""
    trainable: bool
    augmentation: bool
    image_size: str
    batch_size: int
    number_classes: int
    number_epochs: int
    learning_rate: float
    dropout_rate: float
    l2_pentaly_rate: float


@dataclass(frozen=True)
class CallbackConfig:
    """Class to map callback config"""
    callback_path: Path
    #tensorboard_log_path: Path
    model_checkpoint_path: Path


@dataclass
class EvaluationConfig:
    """Class to map evaluation config"""
    test_data_path: Path
    evaluation_score_json_path: Path
    mlflow_uri: str
    track_params: dict
