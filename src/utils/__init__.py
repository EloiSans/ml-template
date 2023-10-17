from torch.optim import Adam

from src.utils.losses import L1MSE

dict_optimizers = {
    "example": Adam
}

dict_loss = {
    "example": L1MSE
}
