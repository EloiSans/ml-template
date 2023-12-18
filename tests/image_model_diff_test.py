import os
from os.path import join
from argparse import BooleanOptionalAction

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from classic_methods import dict_classic_methods
from config import MainArguments
from src.dataset import dict_dataset
from src.model import dict_model
from src.utils.constants import TRAIN
from src.utils.metrics import MetricCalculatorExample
from tests.utils import get_method, save_image

load_dotenv()


if __name__ == '__main__':
    MainArguments.add_argument("--models", nargs='+', type=str, help="Model list", default=list(dict_model.keys()),
                               choices=dict_model.keys())
    MainArguments.add_argument("--max-images", type=int, default=5)
    MainArguments.add_argument("--grouped", action=BooleanOptionalAction, default=True)
    MainArguments.add_argument("--shuffle", action=BooleanOptionalAction, default=True)
    args = MainArguments.parse_args()

    device = args.train_params['device']
    dataset = dict_dataset[args.dataset](**args.data_params, fold=TRAIN, limit=args.max_images, device=device)
    data_loader = DataLoader(dataset, batch_size=args.max_images, shuffle=args.shuffle,
                             num_workers=os.environ.get('WORKERS', 6))
    save_path = join(os.environ["SAVE_PATH"], args.dataset, "images")
    os.makedirs(save_path, exist_ok=True)

    output_dicts = {}
    input_data = {}
    for model in args.models:
        method = get_method(model, args.dataset, device)
        metrics = MetricCalculatorExample(len(data_loader))
        output_data = dict()
        for batch in data_loader:
            if model in dict_classic_methods.keys():
                input_data = {k: v.to(device).permute(0, 2, 3, 1) for k, v in batch.items()}
                aux = torch.ones(input_data['gt'].shape)
                for i in range(args.max_images):
                    unbatched_data = {k: v[i, :, :, :].numpy() for k, v in input_data.items()}
                    out = method(**unbatched_data)
                    aux[i] = torch.from_numpy(list(out.values())[0])
                input_data = {k: v.permute(0, 3, 1, 2) for k, v in input_data.items()}
                output_data = {k: aux.permute(0, 3, 1, 2) for k, v in out.items()}
            else:
                input_data = {k: v.to(device) for k, v in batch.items()}
                output_data = method(**input_data)

            save_image(save_path, input_data, args.dataset, args.grouped)
            save_image(save_path, output_data, model, args.grouped)
