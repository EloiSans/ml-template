import os
from os.path import join

from dotenv import load_dotenv

from config import MainArguments
from src.base import Experiment
from src.model import dict_model
from tests.utils import get_method

load_dotenv()

if __name__ == '__main__':
    MainArguments.add_argument("--models", nargs='+', type=str, help="Model list", default=list(dict_model.keys()),
                               choices=dict_model.keys())
    args = MainArguments.parse_args()
    device = args.train_params['device']
    save_path = join(os.environ["SAVE_PATH"], args.dataset)
    exp = Experiment(**args.__dict__)

    for model_name in args.models:
        model = get_method(model_name, args.dataset, device)
        exp.test(args.output_path, model)
