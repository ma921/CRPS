import torch
import botorch
from gpytorch.mlls import ExactMarginalLogLikelihood


true_f = botorch.test_functions.Ackley(dim=1)

def test_function(n=20):
    X_train = torch.linspace(-3,3,n).unsqueeze(-1)
    Y_train = true_f(X_train).unsqueeze(-1)
    
    gp_test = botorch.models.SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
    )
    mll = ExactMarginalLogLikelihood(gp_test.likelihood, gp_test)
    botorch.fit.fit_gpytorch_mll(mll)
    return gp_test

def test_points(N=200):
    X_test = torch.linspace(-3,3,N).unsqueeze(-1) 
    Y_test = true_f(X_test).unsqueeze(-1)
    return X_test, Y_test