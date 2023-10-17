import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


class TensorboardWriter:
    def __init__(self, model, tensorboard_logdir, model_name=None, dataset_name=None, n_images=5):
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir=tensorboard_logdir)
        self.n_images = n_images

    def add_metrics(self, metrics, phase, epoch):
        self.writer.add_scalars(f"{phase}/metrics", metrics, global_step=epoch)

    def add_losses(self, train_loss, train_loss_comp, val_loss, val_loss_comp, epoch):
        self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, global_step=epoch)
        self.writer.add_scalars("loss/train", train_loss_comp, global_step=epoch)
        self.writer.add_scalars("loss/val", val_loss_comp, global_step=epoch)

    def add_text(self, title, content, step):
        self.writer.add_text(title, content, step)

    def add_images(self, input_data, output_data):
        n_plots = len(input_data) + len(output_data)
        height = 3
        width = 3
        fig, axs = plt.subplots(self.n_images, n_plots, figsize=(5 * width, self.n_images * height))
        for axs_row, i, k, v in zip(axs, enumerate([input_data.items(), output_data.items()])):
            axs_row[i].imshow(self._scaleminmax(self._image_transform(v)).detach().numpy())
            axs_row[i].set_title(k)

            [ax.axis("off") for ax in axs_row]
        return fig

    @staticmethod
    def _scaleminmax(v, new_min=0, new_max=1):
        v_min, v_max = v.min(), v.max()
        return (v - v_min) / (v_max - v_min) * (new_max - new_min) + new_min

    def _image_transform(self, image):
        # add transforms depending on self.model_name or self.dataset_name
        return image

    def model_info(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.writer.add_text("model info", f"Trainable params: {trainable_params}\tTotal params: {total_params}")

    def close(self):
        self.writer.close()


class FileWriter:
    def __init__(self, model_name, dataset_name, output_path):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_path = self._get_output_path(output_path)
        self.csv_path = join(output_path, f'{dataset_name}_metrics.csv')

    @staticmethod
    def _get_output_path(output_path):
        path = join(output_path, 'results')
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            pass
        finally:
            return path

    def add_images(self, input_data, output_data):
        for k, v in input_data.items():
            save_image(self._image_transform(v), join(self.output_path, f'{self.model_name}_{k}.png'))

        for k, v in output_data.items():
            save_image(self._image_transform(v), join(self.output_path, f'{self.model_name}_{k}.png'))

    def add_metrics_to_csv(self, metrics):
        for k, v in metrics.items():
            if type(v) in [int, float]:
                metrics[k] = np.array(v).mean()

        self._write_row(metrics)

    def _write_row(self, metrics):
        write_mode = 'a' if os.path.exists(self.csv_path) else 'w'
        if not os.path.exists(self.csv_path):
            os.makedirs(self.output_path, exist_ok=True)
        with open(self.csv_path, write_mode) as f:
            w = csv.DictWriter(f, metrics.keys())
            if write_mode == 'w':
                w.writeheader()
            w.writerow(metrics)

    def _image_transform(self, image):
        # add transforms depending on self.model_name or self.dataset_name
        return image
