import pytest
import numpy as np
from torch import from_numpy, allclose
from due.networks.fcn import osgnet

# Configuration dictionary with example values
config = {
    "problem_dim": 2,
    "memory": 0,
    "dtype": "double",  # Test with double precision
    "depth": 2,
    "width": 16,
    "activation": "tanh",
    "seed": 0
}

@pytest.fixture
def model():
    if config["dtype"] == "double":
        vmin = -1. * np.ones((1,config["problem_dim"])).astype("float64")
        vmax = np.ones((1,config["problem_dim"])).astype("float64")
        tmin = 0.5
        tmax = 1.5
    else:
        vmin = -1. * np.ones((1,config["problem_dim"])).astype("float32")
        vmax = np.ones((1,config["problem_dim"])).astype("float32")
        tmin = 0.5
        tmax = 1.5
    return osgnet(vmin, vmax, tmin, tmax, config, multiscale=False)

def test_osgnet_output_shape(model):
    batch_size = 4
    input_dim = config["problem_dim"] + 1

    # Generate random input with matching dtype
    if config["dtype"] == "double":
        x = np.random.normal(size=(batch_size, input_dim)).astype("float64")
    else:
        x = np.random.normal(size=(batch_size, input_dim)).astype("float32")
    x = from_numpy(x)
    # Forward pass
    output = model(x)

    # Check if output shape matches the expected shape (batch_size, problem_dim)
    assert output.shape == (batch_size, config["problem_dim"]), \
        f"Expected output shape {(batch_size, config['problem_dim'])}, but got {output.shape}"

def test_osgnet_forward_consistency(model):
    if config["dtype"] == "double":
        x = np.random.normal(size=(4, (config["problem_dim"] + 1))).astype("float64")
    else:
        x = np.random.normal(size=(4, (config["problem_dim"] + 1))).astype("float32")
    x = from_numpy(x)
    # First forward pass
    output1 = model(x)
    # Second forward pass
    output2 = model(x)
    # Ensure consistency of outputs for the same input
    assert allclose(output1, output2), \
        "Forward pass is inconsistent for the same input"

def test_predict_function(model):
    device = "cpu"
    N = 2  # Number of trajectories
    if config["dtype"] == "double":
        x = np.random.normal(size=(N, config["problem_dim"])).astype("float64")
    else:
        x = np.random.normal(size=(N, config["problem_dim"])).astype("float32")
    dt = np.random.uniform(size=(1,5))
    steps = dt.shape[1]
    # Run predict method
    output = model.predict(x, dt, device)

    # Check if output shape matches expected shape (N, problem_dim, steps+memory+1)
    assert output.shape == (N, config["problem_dim"], steps + config["memory"] + 1), \
        f"Expected output shape {(N, config['problem_dim'], steps + config['memory'] + 1)}, but got {output.shape}"

def test_osgnet_seed_reproducibility():
    if config["dtype"] == "double":
        vmin = -1. * np.ones((1,config["problem_dim"])).astype("float64")
        vmax = np.ones((1,config["problem_dim"])).astype("float64")
        tmin = 0.5
        tmax = 1.5
    else:
        vmin = -1. * np.ones((1,config["problem_dim"])).astype("float32")
        vmax = np.ones((1,config["problem_dim"])).astype("float32")
        tmin = 0.5
        tmax = 1.5
    # Initialize two models with the same seed and config
    model1 = osgnet(vmin, vmax, tmin, tmax, config, multiscale=False)
    model2 = osgnet(vmin, vmax, tmin, tmax, config, multiscale=False)

    # Random input with matching dtype
    if config["dtype"] == "double":
        x = np.random.normal(size=(4, (config["problem_dim"] + 1))).astype("float64")
    else:
        x = np.random.normal(size=(4, (config["problem_dim"] + 1))).astype("float32")
    x = from_numpy(x)

    # Forward pass with both models
    output1 = model1(x)
    output2 = model2(x)

    # Check if outputs from both models are the same
    assert allclose(output1, output2), "Outputs are different for the same seed and input."