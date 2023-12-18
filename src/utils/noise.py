import torch
import numpy as np


def add_gaussian_noise(tensor, std_noise, **kwargs):
    return tensor + torch.normal(0, std_noise, size=tensor.shape)


def add_salt_and_pepper_noise(tensor, s_vs_p=0.5, amount=0.04, **kwargs):
    ch, width, height, = tensor.shape

    out = tensor.clone()
    # Salt
    num_salt = int(s_vs_p * amount * width * height)
    cords_h, cords_w = np.random.randint(0, height - 1, num_salt), np.random.randint(0, width - 1, num_salt)
    out[:, cords_h, cords_w] = 1.

    # Pepper
    num_pepper = int((1.-s_vs_p) * amount * width * height)
    cords_h, cords_w = np.random.randint(0, height - 1, num_pepper), np.random.randint(0, width - 1, num_pepper)
    out[:, cords_h, cords_w] = 0.
    return out


def add_poisson_noise(tensor, **kwargs):
    """
    Agrega ruido Poisson a un tensor de imagen.

    Parameters:
    - image_tensor (torch.Tensor): Tensor de imagen al que se le agregará ruido Poisson.

    Returns:
    - torch.Tensor: Tensor de imagen con ruido Poisson.
    """
    vals = torch.unique(tensor)
    vals = 2 ** torch.ceil(torch.log2(torch.tensor(len(vals), dtype=torch.float)))
    noisy_tensor = torch.poisson(tensor * vals) / vals.float()
    return noisy_tensor


def add_noise(tensor, noise='gaussian', noise_params=None, device='cpu'):
    noise_params = noise_params if noise_params is not None else {}

    noise_dict = {'gaussian': add_gaussian_noise,
                  's&p': add_salt_and_pepper_noise,
                  'poisson': add_poisson_noise}

    tensor = noise_dict[noise](tensor, **noise_params)

    return tensor.to(device)
