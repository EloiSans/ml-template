from torch.nn import Module, L1Loss, MSELoss


class LossExample(Module):
    def __init__(self, alpha, **kwargs):
        super(LossExample, self).__init__()
        self.alpha = alpha
        self.l1 = L1Loss()
        self.mse = MSELoss()

    def forward(self, **kwargs):
        pred = kwargs['pred']
        target = kwargs['target']
        mse = self.mse(target, pred)
        l1 = self.alpha * self.l1(pred, target)
        loss = l1 + mse

        return loss, dict(l1=l1.item(), mse=mse.item())
