import argparse

from dotenv import load_dotenv

from config import ModelParser, DataParser, LossParser, OptimParser, TrainParser, MainArguments
from src.base import Experiment
from src.dataset import dict_dataset
from src.loss import dict_loss
from src.model import dict_model
from src.utils.optimizers import dict_optimizer

load_dotenv()

if __name__ == "__main__":
    MainArguments.add_argument('method', choices=['train', 'test', 'classic_methods'])
    args = MainArguments.parse_args()
    exp = Experiment(**args.__dict__)
    if args.method == 'train':
        exp.train()
    elif args.method == 'test':
        exp.test(args.output_path)
    elif args.method == 'classic_methods':
        exp.classical_methods(args.output_path)
