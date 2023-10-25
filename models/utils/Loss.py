import torch

class RMSELoss(torch.nn.Module):
    """ RMSE loss for AutoRec training """
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat,y))