import argparse
import gc
import os
from datetime import datetime as dt

import numpy as np
import torch
from torch.nn.functional import fold, unfold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import dict_dataset
from src.model import dict_model
from src.utils import dict_optimizers, dict_loss
from src.utils.constants import TRAIN, VAL
from src.utils.metrics import MetricCalculator
from src.utils.visualization import TensorboardWriter, FileWriter

gc.collect()
torch.cuda.empty_cache()


class Experiment:
    def __init__(self, dataset: str, model: str, optimizer: str, loss: str,
                 data_params: dict, model_params: dict, loss_params: dict, optim_params: dict,
                 max_epochs: int = 1000, batch_size: int = 10, resume_path: str = None, metric_track_key: str = None,
                 metric_track_mode: str = None, **kwargs):

        self.dataset_name = dataset
        self.dataset = dict_dataset[dataset]
        self.data_params = data_params

        self.model_name = model
        self.model = dict_model[model]
        self.model_params = model_params

        self.optimizer = dict_optimizers[optimizer]
        self.optim_params = optim_params

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = dict_loss[loss](**loss_params, device=self.device)
        self.eval_n = max(int(max_epochs * (os.environ.get('EVAL_FREQ', 100) / 100)), 1)
        self.save_path = os.path.join(os.environ["SAVE_PATH"], dataset, dt.now().strftime("%Y-%m-%d"), model)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.resume_path = resume_path

        self.writer = None
        self.f_writer = FileWriter

        self.metric_calculator = MetricCalculator
        self.metric_track_key = metric_track_key
        self.metric_track_mode = metric_track_mode
        assert self.metric_track_mode in ['min', 'max']
        self.best_metric = 0 if self.metric_track_mode == 'max' else 1000

        self.workers = int(os.environ.get('WORKERS', 6))

        seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        self.writer = TensorboardWriter(self.model, self.save_path)

        # training and validation data loaders
        dataset_train = self.dataset(**self.data_params, fold=TRAIN)
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

        dataset_val = self.dataset(**self.data_params, fold=VAL)
        val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=self.workers)

        model, start_epoch = self._init_model()

        model = model.float()
        model.to(self.device)

        optimizer = self.optimizer(model.parameters(), **self.optim_params)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=30)

        for epoch in range(start_epoch, self.max_epochs):
            model.train()
            train_loss, train_loss_comp, train_metrics = self._main_phase(train_loader, TRAIN, epoch)
            with torch.no_grad():
                model.eval()
                val_loss, val_loss_comp, val_metrics = self._main_phase(val_loader, VAL, epoch)

            scheduler.step(val_loss)

            if epoch % self.eval_n == 0:
                self.writer.add_losses(train_loss, train_loss_comp, val_loss, val_loss_comp, epoch)

            if self._is_best_metric(val_metrics):
                self.writer.add_metrics(val_metrics, VAL, epoch)
                self._save_model(model, 'best', epoch)

            self._save_model(model, 'last', epoch)
        self.writer.close()

    def eval(self, output_path):
        self.writer = FileWriter(self.model_name, self.dataset_name, output_path)

        dataset = self.dataset(**self.data_params, fold=VAL)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.workers)

        model, _ = self._init_model()
        total_parameters = sum(p.numel() for p in model.parameters())

        for param in model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            model.eval()
            loss, loss_comp, metrics = self._main_phase(data_loader, VAL)

        csv_save_dict = dict(**metrics, name=self.model_name, params=total_parameters)

        self.writer.add_metrics_to_csv(csv_save_dict)
        return

    def _save_model(self, model, version, epoch: int = None):
        try:
            os.makedirs(self.save_path + '/ckpt/')
        except FileExistsError:
            pass
        save_path = self.save_path + f'/ckpt/weights_{version}.pth'
        ckpt = {'experiment': self, 'model': model, 'epoch': epoch}
        torch.save(ckpt, save_path)

    def _init_model(self):
        if self.resume_path is not None:
            ckpt = torch.load(self.resume_path, map_location=self.device)
            model = ckpt['model']
            start_epoch = ckpt.get('epoch', 0) + 1
        else:
            model = self.model(**self.model_params)
            start_epoch = 0

        return model, start_epoch

    @staticmethod
    def _do_patches(data, ps):
        data_shape = data.size
        assert data_shape(2) % ps == 0 and data_shape(3) % ps == 0
        patches = unfold(data, kernel_size=ps, stride=ps)
        patch_num = patches.size(2)
        patches = patches.permute(0, 2, 1).view(data_shape(0), -1, data_shape(1), ps, ps)
        return torch.reshape(patches, (data_shape(0) * patch_num, data_shape(1), ps, ps))

    @staticmethod
    def _undo_patches(data, n, w, h, ps):
        patches = data.reshape(n, data.size(0), data.size(1), ps, ps)
        patches = patches.view(n, data.size(0), data.size(1) * ps * ps).permute(0, 2, 1)
        return fold(patches, (w, h), kernel_size=ps, stride=ps)

    def _main_phase(self, data_loader, phase, epoch=None):
        metrics = self.metric_calculator(len(data_loader))
        epoch_loss = []
        epoch_loss_components = dict(dice=[], data_term=[], radio_term=[])
        with tqdm(enumerate(data_loader), total=len(data_loader), leave=True) as pbar:
            for idx, batch in pbar:
                self.optimizer.zero_grad()

                input_data = {k: v.to(self.device) for k, v in batch.items()}

                output_data = self.model(**input_data)

                loss, loss_components = self.loss(**input_data, **output_data)
                self.writer.add_images(input_data, output_data)

                epoch_loss.append(loss.item())
                for k in epoch_loss_components.keys():
                    epoch_loss_components[k].append(loss_components[k].cpu().detach().numpy())

                if phase == TRAIN:
                    loss.backward()
                    self.optimizer.step()

                metrics.add_metrics(**input_data, **output_data)

                if epoch:
                    pbar.set_description(f'Epoch: {epoch + 1}; {phase} Loss {np.array(epoch_loss).mean():.6f}')

        for k in epoch_loss_components.keys():
            epoch_loss_components[k] = np.array(epoch_loss_components[k]).mean()

        epoch_loss = np.array(epoch_loss).mean()
        if epoch % self.eval_n == 0:
            self.writer.add_metrics(metrics, phase, epoch)

        return epoch_loss, epoch_loss_components, metrics

    def load_from_dict(self, **ckpt):
        for k, v in ckpt.items():
            setattr(self, k, v)

    def _is_best_metric(self, metric_dict):
        value = metric_dict[self.metric_track_key]
        if self.metric_track_mode == 'min':
            result = self.best_metric > value
            self.best_metric = min(self.best_metric, value)
        else:
            result = self.best_metric < value
            self.best_metric = max(self.best_metric, value)
        return result


class ParseKwargs(argparse.Action):
    CHOICES = dict()

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.CHOICES)
        for value in values:
            key, value = value.split('=')
            if self.CHOICES and key not in self.CHOICES.keys():
                print(f"{parser.prog}: error: argument {option_string}: invalid choice: '{key}' (choose from {list(self.CHOICES.keys())})")
            else:
                getattr(namespace, self.dest)[key] = self._parse(value)

    @staticmethod
    def _parse(data):
        try:
            return int(data)
        except ValueError:
            pass
        try:
            return float(data)
        except ValueError:
            pass
        return data