from torch.nn import Module, L1Loss, MSELoss


class L1MSE(Module):
    def __init__(self, alpha, **kwargs):
        super(L1MSE, self).__init__()
        self.alpha = alpha
        self.l1 = L1Loss()
        self.mse = MSELoss()

    def forward(self, pred, target):
        mse = self.mse(target, pred)
        l1 = self.alpha * self.l1(pred, target)
        loss = l1 + mse

        return loss, dict(l1=l1.item(), mse=mse.item())
