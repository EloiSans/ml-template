import argparse

from src.base import ParseKwargs
from src.dataset import dict_dataset
from src.loss import dict_loss
from src.model import dict_model
from src.utils.optimizers import dict_optimizer


class DataParser(ParseKwargs):
    CHOICES = {
        "crop_size": 32,
        "upscale_factor": 4,
        "noise": None
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
        "max_epochs": 1,
        "batch_size": 1,
        "metric_track_key": 'psnr',
        "metric_track_mode": "min",
        "device": "cpu"
    }


class NoiseParser(ParseKwargs):
    CHOICES = {
        "std_noise": 0.01,
        "s_vs_p": 0.5,
        "amount": 0.04
    }


MainArguments = argparse.ArgumentParser(description="Add new parameters in config.py")
MainArguments.add_argument("--dataset", type=str, help="Dataset name", default=list(dict_dataset.keys())[0], choices=dict_dataset.keys())
MainArguments.add_argument("--model", type=str, help="Model name", default=list(dict_model.keys())[0], choices=dict_model.keys())
MainArguments.add_argument("--loss", type=str, help="Loss name", default=list(dict_loss.keys())[0], choices=dict_loss.keys())
MainArguments.add_argument("--optimizer", type=str, help="Optimizer name", default=list(dict_optimizer.keys())[0], choices=dict_optimizer.keys())

MainArguments.add_argument("--data-params", nargs='*', action=DataParser, default=DataParser.CHOICES, help="Dataset parameters")
MainArguments.add_argument("--model-params", nargs='*', action=ModelParser, default=ModelParser.CHOICES, help="Model parameters")
MainArguments.add_argument("--loss-params", nargs='*', action=LossParser, default=LossParser.CHOICES, help="Loss parameters")
MainArguments.add_argument("--optim-params", nargs='*', action=OptimParser, default=OptimParser.CHOICES, help="Optim parameters")
MainArguments.add_argument("--train-params", nargs='*', action=TrainParser, default=TrainParser.CHOICES, help="Train parameters")
MainArguments.add_argument("--noise-params", nargs='*', action=NoiseParser, default=NoiseParser.CHOICES, help="Noise parameters")

MainArguments.add_argument("--resume-path", type=str, help="Resume path")
MainArguments.add_argument("--output-path", type=str, default="runs/", help="Output path")
