import pytest
import numpy as np
from scipy.io import savemat, loadmat
from due.datasets.ode import ode_dataset_osg
import os
@pytest.fixture
def setup_data():
    config = {
        "problem_dim": 2,
        "nbursts": 5,
        "multiscale": False
    }
    dataset = ode_dataset_osg(config)
    
    train_data = {
        "dt": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "trajectories": np.random.rand(1, 2, 5)
    }
    savemat("train_data.mat", train_data)
    
    test_data = {
        "dt": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "trajectories": np.random.rand(1, 2, 5)
    }
    savemat("test_data.mat", test_data)
    
    return dataset

def test_load(setup_data):
    dataset = setup_data
    trainX, trainY, dt, vmin, vmax, tmin, tmax = dataset.load("train_data.mat", None)
    assert trainX.shape[1] == dataset.problem_dim + 1
    assert trainY.shape[1] == dataset.problem_dim
    assert len(dt) == 1
    assert vmin.shape[1] == dataset.problem_dim
    assert vmax.shape[1] == dataset.problem_dim
    assert isinstance(tmin, float)
    assert isinstance(tmax, float)

def test_load_with_test_data(setup_data):
    dataset = setup_data
    trainX, trainY, test_data, dt, vmin, vmax, tmin, tmax = dataset.load("train_data.mat", "test_data.mat")
    assert trainX.shape[1] == dataset.problem_dim + 1
    assert trainY.shape[1] == dataset.problem_dim
    assert test_data.shape[1] == dataset.problem_dim
    assert len(dt) == 1
    assert vmin.shape[1] == dataset.problem_dim
    assert vmax.shape[1] == dataset.problem_dim
    assert isinstance(tmin, float)
    assert isinstance(tmax, float)
    os.remove("train_data.mat")
    os.remove("test_data.mat")