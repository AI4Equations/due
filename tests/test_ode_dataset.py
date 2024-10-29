import pytest
import numpy as np
from due.datasets.ode import ode_dataset

# Define configuration for the test
config = {
    "problem_dim": 3,
    "memory": 5,
    "multi_steps": 2,
    "nbursts": 10,
    "dtype": "double"
}

@pytest.fixture
def dataset():
    return ode_dataset(config)

def test_initialization(dataset):
    assert dataset.problem_dim == config["problem_dim"]
    assert dataset.memory_steps == config["memory"]
    assert dataset.multi_steps == config["multi_steps"]
    assert dataset.nbursts == config["nbursts"]
    assert dataset.dtype == config["dtype"]

def test_load_dataset(dataset, tmp_path):
    # Create mock .mat files for train and test
    train_path = tmp_path / "train_data.mat"
    test_path = tmp_path / "test_data.mat"

    # Mock data structure
    data = {"trajectories": np.random.rand(20, config["problem_dim"], 50)}
    from scipy.io import savemat
    savemat(train_path, data)
    savemat(test_path, data)

    # Test loading and normalization
    trainX, trainY, vmin, vmax = dataset.load(train_path, None)

    # Check shapes of trainX and trainY based on config parameters
    expected_shape_X = (config["nbursts"] * 20, (config["memory"] + 1) * config["problem_dim"])
    expected_shape_Y = (config["nbursts"] * 20, config["problem_dim"], config["multi_steps"] + 1)
    assert trainX.shape == expected_shape_X
    assert trainY.shape == expected_shape_Y

def test_normalize(dataset):
    # Create a random dataset
    data = np.random.rand(10, config["problem_dim"], 50)

    # Normalize data
    normalized_data, vmin, vmax = dataset.normalize(data)

    # Check normalization range
    assert np.all(normalized_data >= -1) and np.all(normalized_data <= 1)
