import pytest
import numpy as np
from scipy.io import savemat
from due.datasets.pde import pde_dataset

@pytest.fixture
def config():
    return {
        "problem_type": "1d_regular",
        "memory": 3,
        "multi_steps": 2,
        "nbursts": 10,
        "dtype": "single"
    }

@pytest.fixture
def dummy_data_1d():
    # Create dummy 1D regular data with a shape (N, L, D, T)
    return {
        "coordinates": np.linspace(0, 1, 100).reshape(-1, 1),
        "trajectories": np.random.rand(5, 100, 3, 50)
    }

@pytest.fixture
def dummy_data_2d():
    # Create dummy 2D regular data with a shape (N, H, W, D, T)
    return {
        "coordinates": np.random.rand(50, 50, 2),
        "trajectories": np.random.rand(5, 50, 50, 3, 40)
    }

def test_initialization(config):
    dataset = pde_dataset(config)
    assert dataset.problem_type == config["problem_type"]
    assert dataset.memory_steps == config["memory"]
    assert dataset.dtype == config["dtype"]
    assert dataset.nbursts == config["nbursts"]

def test_load_1d_regular(dummy_data_1d, config, tmp_path, monkeypatch):
    # Create a temporary .mat file with dummy 1D data
    file_path_train = tmp_path / "dummy_1d.mat"
    savemat(file_path_train, dummy_data_1d)

    dataset = pde_dataset(config)
    
    # Mock loadmat to load from our fixture directly
    monkeypatch.setattr("due.datasets.pde.loadmat", lambda x: dummy_data_1d)
    
    trainX, trainY, coords = dataset.load(file_path_train, None)
    
    assert trainX.shape[-1] == config["memory"] + 1
    assert trainY.shape[-1] == config["multi_steps"] + 1
    assert coords.shape[1] == 1

def test_load_2d_regular(dummy_data_2d, config, tmp_path, monkeypatch):
    # Adjust configuration for 2D data
    config["problem_type"] = "2d_regular"

    # Create a temporary .mat file with dummy 2D data
    file_path_train = tmp_path / "dummy_2d.mat"
    savemat(file_path_train, dummy_data_2d)

    dataset = pde_dataset(config)
    
    # Mock loadmat to load from our fixture directly
    monkeypatch.setattr("due.datasets.pde.loadmat", lambda x: dummy_data_2d)
    
    trainX, trainY, coords = dataset.load(file_path_train, None)
    
    assert trainX.shape[1] == 50  # H dimension
    assert trainX.shape[2] == 50  # W dimension
    assert coords.shape[-1] == 2  # Coordinate grid (H, W, 2)

def test_memory_steps_limit(config, dummy_data_1d, tmp_path, monkeypatch):
    config["nbursts"] = 1000  # Excessively high to trigger the internal limit adjustment

    # Mock loadmat and use 1D data for testing
    monkeypatch.setattr("due.datasets.pde.loadmat", lambda x: dummy_data_1d)
    
    dataset = pde_dataset(config)
    dataset.load(tmp_path / "dummy_file.mat", None)
    
    assert dataset.nbursts < 1000, "nbursts should be limited by time dimension - memory_steps - multi_steps - 1"

def test_load_exceptions(config, tmp_path, monkeypatch):
    # Test loading with invalid data format
    file_path_train = tmp_path / "dummy_invalid.mat"
    savemat(file_path_train, {"invalid_key": np.array([1, 2, 3])})

    dataset = pde_dataset(config)
    
    with pytest.raises(ValueError, match="Please name your dataset as trajectories."):
        dataset.load(file_path_train, None)
    
    # Test with unsupported problem type
    config["problem_type"] = "3d_regular"
    savemat(file_path_train, {"trajectories": np.ones((1, 2, 3,20)), "coordinates": np.ones((890,1))})
    with pytest.raises(ValueError, match="1D and 2D data collected on either uniform grids or unstructured meshes are supported. Make sure that your dataset is correctly organized. 3D problems are not yet supported"):
        dataset = pde_dataset(config)
        dataset.load(file_path_train, None)
