import os
from os.path import join

import torch
from torchvision.utils import save_image as si

from classic_methods import dict_classic_methods


def get_resume_path(model, dataset):
    base_path = os.path.join(os.environ["SAVE_PATH"], dataset, model)
    date = sorted([dir for dir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, dir))])[-1]
    final_path = "ckpt/weights_last.pth"
    return os.path.join(os.environ["SAVE_PATH"], dataset, model, date, final_path)


def get_method(model, dataset, device):
    if model in dict_classic_methods.keys():
        return dict_classic_methods[model]
    else:
        resume_path = get_resume_path(model, dataset)
        ckpt = torch.load(resume_path, map_location=device)
        model = ckpt['model']
        model = model.float()
        model.to(device)
        return model


def save_image(save_path, data, alias, grouped):
    if grouped:
        for k, v in data.items():
            si(v, join(save_path, f'{alias}_{k}.png'))

    else:
        for k, v in data.items():
            for id, img in enumerate(v):
                si(img, join(save_path, f'{alias}_{id}_{k}.png'))
