import os
import tarfile
from urllib.request import urlopen
from os import makedirs, remove, listdir
from os.path import exists, join, basename

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def download_bsd300():
    dest = os.environ['DATASET_PATH']
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
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
    def __init__(self, crop_size, upscale_factor, fold):
        super(DatasetExample, self).__init__()
        image_dir = join(os.environ['DATASET_PATH'], "BSDS300/images")
        self.image_filenames = [join(image_dir, fold, x) for x in listdir(image_dir) if is_image_file(x)]
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

        self.input_transform = self._input_transform(crop_size, upscale_factor)
        self.target_transform = self._target_transform(crop_size)

        download_bsd300()

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

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
