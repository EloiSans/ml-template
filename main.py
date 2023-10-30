import argparse

from config import ModelParser, DataParser, LossParser, OptimParser, TrainParser
from src.dataset import dict_dataset
from src.loss import dict_loss
from src.model import dict_model
from src.utils.optimizers import dict_optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion with unfolding")
    parser.add_argument("--dataset", type=str, help="Dataset name", default=list(dict_dataset.keys())[0], choices=dict_dataset.keys())
    parser.add_argument("--model", type=str, help="Model name", default=list(dict_model.keys())[0], choices=dict_model.keys())
    parser.add_argument("--loss", type=str, help="Loss name", default=list(dict_loss.keys())[0], choices=dict_loss.keys())
    parser.add_argument("--optimizer", type=str, help="Optimizer name", default=list(dict_optimizer.keys())[0], choices=dict_optimizer.keys())

    parser.add_argument("--data_params", nargs='*', action=DataParser, help="Dataset parameters")
    parser.add_argument("--model_params", nargs='*', action=ModelParser, help="Model parameters")
    parser.add_argument("--loss_params", nargs='*', action=LossParser, help="Loss parameters")
    parser.add_argument("--optim_params", nargs='*', action=OptimParser, help="Optim parameters")
    parser.add_argument("--train_params", nargs='*', action=TrainParser, help="Train parameters")

    parser.add_argument("--resume_path", type=str, help="Train parameters")

    args = parser.parse_args()
    print(args.__dict__['model_params'])
    args_dict = args.__dict__
