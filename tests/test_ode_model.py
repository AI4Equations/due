import pytest
import numpy as np
from due.utils import *
from due.networks.fcn import resnet
from due.models import ODE 

# Configuration dictionary with example values
config_net = {
    "problem_dim": 5,
    "memory": 0,
    "dtype": "single",  # Test with single precision
    "depth": 2,
    "width": 16,
    "activation": "relu",
    "seed": 0
}

@pytest.fixture
def mock_network():
    if config_net["dtype"] == "double":
        vmin = -1. * np.ones((1,config_net["problem_dim"],1)).astype("float64")
        vmax = np.ones((1,config_net["problem_dim"],1)).astype("float64")
    else:
        vmin = -1. * np.ones((1,config_net["problem_dim"],1)).astype("float32")
        vmax = np.ones((1,config_net["problem_dim"],1)).astype("float32")
    return resnet(vmin, vmax, config_net)

@pytest.fixture
def config():
    return {
        "seed": 42,
        "device": "cpu",
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "scheduler": "None",
        "loss": "mse",
        "save_path": "./test_save",
        "verbose": 1,
    }

@pytest.fixture
def dummy_data():
    trainX = np.random.rand(10, 5).astype(np.float32)
    trainY = np.random.rand(10, 5, 10).astype(np.float32)
    return trainX, trainY

@pytest.fixture
def ode_model(config, dummy_data, mock_network):
    trainX, trainY = dummy_data
    return ODE(trainX, trainY, mock_network, config)


# Test initialization
def test_initialization(ode_model, config):
    assert ode_model.device == config["device"]
    assert ode_model.nepochs == config["epochs"]
    assert ode_model.bsize == config["batch_size"]
    assert ode_model.lr == config["learning_rate"]

# Test training process
def test_training_process(ode_model):
    ode_model.train()
    # Check that training history has been populated
    assert ode_model.hist.shape[0] == ode_model.nepochs
