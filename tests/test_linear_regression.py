import pytest
import due
import os
import numpy as np
from scipy.io import savemat, loadmat


def test_linear_regression():
    # generate data set for linear regression
    X = np.random.uniform(size=(10000,1,1))
    Y = 2. * X + 1. # y=2x+1
    savemat("test_train.mat", mdict={"trajectories":np.concatenate((X,Y), axis=-1)})
    conf_data = {
        "problem_dim": 1,
        "memory": 0,
        "multi_steps": 0,
        "nbursts": 1,
        "dtype": "double"
    }
    # Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles
    data_loader = due.datasets.ode.ode_dataset(conf_data)
    trainX, trainY, vmin, vmax = data_loader.load("test_train.mat", None)
    # construct an affine model
    conf_net = {
        "problem_dim": 1,
        "memory": 0,
        "dtype": "double",  # Test with double precision
        "seed": 0
    }
    mynet = due.networks.fcn.affine(vmin, vmax, conf_net)

    ### do linear regression
    conf_train = {
            "seed": 42,
            "device": "cpu",
            "epochs": 200,
            "batch_size": 1000,
            "learning_rate": 1e-2,
            "optimizer": "adam",
            "scheduler": "None",
            "loss": "mse",
            "save_path": "./test_save",
            "verbose": 100,
        }
    model = due.models.ODE(trainX, trainY, mynet, conf_train)
    model.train()
    assert np.allclose(mynet.mDMD.weight.detach().cpu().numpy(), np.array([[2.0]]), rtol=1e-04, atol=1e-08)
    assert np.allclose(mynet.mDMD.bias.detach().cpu().numpy(), np.array([5.0/3.0]), rtol=1e-04, atol=1e-08)
    os.remove("test_train.mat")
