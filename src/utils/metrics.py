from torchmetrics.functional.image import error_relative_global_dimensionless_synthesis as ERGAS
from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM


class MetricCalculatorExample(object):
    def __init__(self, dataset_len, sampling_factor=4):
        super().__init__()

        self.len = dataset_len
        self.dict = {'ergas': 0, 'psnr': 0, 'ssim': 0}
        self.sampling_factor = sampling_factor

    def add_metrics(self, **kwargs):
        pred = kwargs['pred']
        target = kwargs['target']

        psnr = PSNR(pred, target).item()
        ergas = ERGAS(pred, target, self.sampling_factor).item()
        ssim = SSIM(pred, target, data_range=1.).item()

        N = pred.shape[0]

        self.dict['ergas'] += N * ergas / self.len
        self.dict['psnr'] += N * psnr / self.len
        self.dict['ssim'] += N * ssim / self.len
