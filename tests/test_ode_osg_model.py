import pytest
import numpy as np
import torch
from due.utils import *
from due.networks.fcn import osgnet
from due.models import ODE, ODE_osg
import os

config = {
        "dtype": "double",
        "sg_pairing": 1,
        "sg_weight": 0.5,
        "device": "cpu",
        "optimizer": "Adam",
        "scheduler": "cosine",
        "save_path": ".",
        "verbose": 1,
        "seed": 0,
        "problem_dim": 2,
        "memory": 0,
        "depth": 2,
        "width": 10,
        "activation": "gelu",
        "epochs": 2,
        "batch_size": 1,
        "learning_rate": 0.001,
        "loss": "mse",
        "valid": 0
        }
def MockNetwork():
    if config["dtype"] == "double":
        vmin = -1. * np.ones((1,config["problem_dim"],1)).astype("float64")
        vmax = np.ones((1,config["problem_dim"],1)).astype("float64")
        tmin = 0.5
        tmax = 1.5
    else:
        vmin = -1. * np.ones((1,config["problem_dim"],1)).astype("float32")
        vmax = np.ones((1,config["problem_dim"],1)).astype("float32")
        tmin = 0.5
        tmax = 1.5
    return osgnet(vmin, vmax, tmin, tmax, config, multiscale=False)

@pytest.fixture
def setup_data():
    trainX = np.random.randn(10, 3)
    trainY = np.random.randn(10, 2)
    osg_data = (np.random.rand(10, config["sg_pairing"], config["problem_dim"]), np.random.rand(10, config["sg_pairing"], 3))
    network = MockNetwork()
    
    return trainX, trainY, osg_data, network, config

def test_osg_regularization_with_data(setup_data):
    trainX, trainY, osg_data, network, config = setup_data
    model = ODE_osg(trainX, trainY, osg_data, network, config)
    
    assert model.mynet.vmin is not None
    assert model.mynet.vmax is not None
    assert model.mynet.tmin is not None
    assert model.mynet.tmax is not None

def test_osg_regularization_without_data(setup_data):
    trainX, trainY, _, network, config = setup_data
    osg_data = None
    model = ODE_osg(trainX, trainY, osg_data, network, config)
    
    assert model.vmin is network.vmin
    assert model.vmax is network.vmax
    assert model.tmin is network.tmin
    assert model.tmax is network.tmax

def test_osg_data_shape(setup_data):
    trainX, trainY, osg_data, network, config = setup_data
    model = ODE_osg(trainX, trainY, osg_data, network, config)
    
    assert len(osg_data[0].shape) == 3
    assert len(osg_data[1].shape) == 3

def test_train_function(setup_data):
    trainX, trainY, osg_data, network, config = setup_data
    model = ODE_osg(trainX, trainY, osg_data, network, config)
    
    model.train()
    assert model.hist is not None
    assert model.hist.shape == (config["epochs"], 3)
    os.remove("model")