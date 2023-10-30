
from src.base import ParseKwargs


class DataParser(ParseKwargs):
    CHOICES = {
        "str": "a",
        "int": 1,
        "float": 1.5,
        "e": 1.e3
    }


class ModelParser(ParseKwargs):
    CHOICES = {
        "str": "a",
        "int": 1,
        "float": 1.5,
        "e": 1.e3
    }


class LossParser(ParseKwargs):
    CHOICES = {
        "str": "a",
        "int": 1,
        "float": 1.5,
        "e": 1.e3
    }


class OptimParser(ParseKwargs):
    CHOICES = {
        "lr": 1e-3,
        "scheduler": 'ReduceLROnPlateau',
        "float": 1.5,
        "e": 1.e3
    }


class TrainParser(ParseKwargs):
    CHOICES = {
        "max_epochs": 1000,
        "batch_size": 1,
        "metric_track_key": 1.5,
        "metric_track_mode": "min"
    }
