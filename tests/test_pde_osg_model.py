import pytest
import numpy as np
import torch
from due.utils import *
from due.networks.fno import osg_fno2d
from due.models import PDE_osg
import os

config = {
    "dtype": "single",
    "sg_pairing": 1,
    "sg_weight": 0.5,
    "device": "cuda",
    "optimizer": "Adam",
    "scheduler": "cosine",
    "save_path": ".",
    "verbose": 1,
    "seed": 0,
    "problem_type": "2d_regular",
    "problem_dim": 2,
    "memory": 0,
    "depth": 2,
    "width": 10,
    "activation": "gelu",
    "modes1": 6,
    "modes2": 6,
    "epochs": 2,
    "batch_size": 1,
    "learning_rate": 0.001,
    "loss": "rel_l2"
}

def MockNetwork():
    vmin = -1. * np.ones((1, 1, 1, config["problem_dim"], 1)).astype("float32")
    vmax = np.ones((1, 1, 1, config["problem_dim"], 1)).astype("float32")
    tmin = 0.5
    tmax = 1.5
    return osg_fno2d(vmin, vmax, tmin, tmax, config, multiscale=False)

@pytest.fixture
def setup_data():
    trainX = np.random.randn(10, 16, 16, config["problem_dim"]+1).astype("float32")
    trainY = np.random.randn(10, 16, 16, config["problem_dim"]).astype("float32")
    coordinates = np.random.rand(16, 16, 2).astype("float32")
    osg_data = (np.random.rand(10, config["sg_pairing"], 16, 16, config["problem_dim"]).astype("float32"), np.random.rand(10, config["sg_pairing"], 16, 16, 3).astype("float32"))
    network = MockNetwork()
    
    return trainX, trainY, osg_data, network, config

def test_osg_regularization_with_data(setup_data):
    trainX, trainY, osg_data, network, config = setup_data
    model = PDE_osg(trainX, trainY, osg_data, network, config)
    
    assert model.mynet.vmin is not None
    assert model.mynet.vmax is not None
    assert model.mynet.tmin is not None
    assert model.mynet.tmax is not None

def test_osg_regularization_without_data(setup_data):
    trainX, trainY, _, network, config = setup_data
    osg_data = None
    model = PDE_osg(trainX, trainY, osg_data, network, config)
    
    assert model.vmin is network.vmin
    assert model.vmax is network.vmax
    assert model.tmin is network.tmin
    assert model.tmax is network.tmax

def test_osg_data_shape(setup_data):
    trainX, trainY, osg_data, network, config = setup_data
    model = PDE_osg(trainX, trainY, osg_data, network, config)
    
    assert len(osg_data[0].shape) == 5
    assert len(osg_data[1].shape) == 5

def test_train_function(setup_data):
    trainX, trainY, osg_data, network, config = setup_data
    model = PDE_osg(trainX, trainY, osg_data, network, config)
    
    model.train()
    assert model.hist is not None
    assert model.hist.shape == (config["epochs"], 1)
    os.remove("model")