
from src.base import ParseKwargs


class DataParser(ParseKwargs):
    CHOICES = {
        "crop_size": 32,
        "upscale_factor": 4
    }


class ModelParser(ParseKwargs):
    CHOICES = {
        "upscale_factor": 4
    }


class LossParser(ParseKwargs):
    CHOICES = {
        "alpha": 0.1
    }


class OptimParser(ParseKwargs):
    CHOICES = {
        "lr": 1e-3,
        "scheduler": 'ReduceLROnPlateau',
    }


class TrainParser(ParseKwargs):
    CHOICES = {
        "max_epochs": 1000,
        "batch_size": 1,
        "metric_track_key": 'psnr',
        "metric_track_mode": "min"
    }
