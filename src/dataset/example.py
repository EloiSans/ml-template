import os
import tarfile
from urllib.request import urlopen
from os import makedirs, remove, listdir
from os.path import exists, join, basename

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from src.utils.noise import add_noise


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def download_bsd300():
    dest = os.environ['DATASET_PATH']
    output_image_dir = join(dest, "BSDS300")

    if not exists(output_image_dir):
        makedirs(dest, exist_ok=True)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


class DatasetExample(Dataset):
    def __init__(self, crop_size, upscale_factor, fold, device, limit=None, noise=None, noise_params=None, **kwargs):
        super(DatasetExample, self).__init__()
        image_dir = join(os.environ['DATASET_PATH'], "BSDS300")
        download_bsd300()
        self.image_filenames = [join(image_dir, fold, x) for x in listdir(join(image_dir, fold)) if is_image_file(x)]
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

        self.input_transform = self._input_transform(crop_size, upscale_factor)
        self.target_transform = self._target_transform(crop_size)
        self.limit = limit
        self.noise = noise
        self.noise_params = noise_params

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        if self.noise:
            input = add_noise(input, self.noise, self.noise_params, )

        return dict(input=input, target=target)

    def __len__(self):
        _len = len(self.image_filenames)
        return _len if self.limit is None else min(_len, self.limit)

    @staticmethod
    def _input_transform(crop_size, upscale_factor):
        return Compose([
            CenterCrop(crop_size),
            Resize(crop_size // upscale_factor),
            ToTensor(),
        ])

    @staticmethod
    def _target_transform(crop_size):
        return Compose([
            CenterCrop(crop_size),
            ToTensor(),
        ])
