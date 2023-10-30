from torchmetrics.functional.image import error_relative_global_dimensionless_synthesis as ERGAS
from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image import spectral_angle_mapper as SAM
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM


class MetricCalculatorExample(dict):
    def __init__(self, dataset_len, sampling_factor=4):
        super().__init__()

        self.len = dataset_len
        self.dict = {'ergas': 0, 'psnr': 0, 'ssim': 0, 'sam': 0}
        self.sampling_factor = sampling_factor

    def add_metrics(self, pred, target):

        psnr = PSNR(pred, target).item()
        ergas = ERGAS(pred, target, self.sampling_factor).item()
        ssim = SSIM(pred, target, data_range=1.).item()
        sam = SAM(pred, target).item()

        N = pred.shape[0]

        self.dict['sam'] += N * sam / self.len
        self.dict['ergas'] += N * ergas / self.len
        self.dict['psnr'] += N * psnr / self.len
        self.dict['ssim'] += N * ssim / self.len

    def __repr__(self):
        for k, v in self.dict.items():
            self.__setitem__(k, v)
            return super().__repr__()
