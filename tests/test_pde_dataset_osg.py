import pytest
import numpy as np
from scipy.io import savemat
from due.datasets.pde import pde_dataset_osg

@pytest.fixture
def setup_data():
    config = {
        "problem_type": "1d_irregular",
        "nbursts": 5,
        "multiscale": False,
        "dtype": "float32"
    }
    dataset = pde_dataset_osg(config)
    
    train_data = {
        "dt": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "trajectories": np.random.rand(1, 5, 2, 5).astype(np.float32),
        "coordinates": np.random.rand(1, 5).astype(np.float32)
    }
    savemat("train_data.mat", train_data)
    
    test_data = {
        "dt": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "trajectories": np.random.rand(1, 5, 2, 5).astype(np.float32),
        "coordinates": np.random.rand(1, 5).astype(np.float32)
    }
    savemat("test_data.mat", test_data)
    
    return dataset

def test_load(setup_data):
    dataset = setup_data
    trainX, trainY, coords, dt, vmin, vmax, tmin, tmax, cmin, cmax = dataset.load("train_data.mat", None)
    assert trainX.shape[2] == 3  # problem_dim + 1
    assert trainY.shape[2] == 2  # problem_dim
    assert len(dt) == 1
    assert vmin.shape[2] == 2  # problem_dim
    assert vmax.shape[2] == 2  # problem_dim


def test_load_with_test_data(setup_data):
    dataset = setup_data
    trainX, trainY, coords, test_data, dt, vmin, vmax, tmin, tmax, cmin, cmax = dataset.load("train_data.mat", "test_data.mat")
    assert trainX.shape[2] == 3  # problem_dim + 1
    assert trainY.shape[2] == 2  # problem_dim
    assert test_data.shape[2] == 2  # problem_dim
    assert len(dt) == 1
    assert vmin.shape[2] == 2  # problem_dim
    assert vmax.shape[2] == 2  # problem_dim
