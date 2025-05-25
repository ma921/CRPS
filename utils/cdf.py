import torch

def iECDF(f, k):
    """
    inverse empirical CDF
    
    f: samples
    k: quantile
    """
    M = len(f)
    sorted = f.squeeze().sort()[0]
    return sorted[int(M * k)]

class ECDF(torch.nn.Module):
    """
    Empirical CDF

    x: samples
    """
    def __init__(self, x, side='right'):
        super(ECDF, self).__init__()

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        if len(x.shape) != 1:
            msg = 'x must be 1-dimensional'
            raise ValueError(msg)

        x = x.sort()[0]
        nobs = len(x)
        y = torch.linspace(1./nobs, 1, nobs, device=x.device)

        self.x = torch.cat((torch.tensor([-torch.inf], device=x.device), x))
        self.y = torch.cat((torch.tensor([0], device=y.device), y))
        self.n = self.x.shape[0]

    def forward(self, time):
        tind = torch.searchsorted(self.x, time, side=self.side) - 1
        return self.y[tind]
