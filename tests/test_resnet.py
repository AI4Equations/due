import pytest
import numpy as np
from torch import from_numpy, allclose
from due.networks.fcn import resnet

# Configuration dictionary with example values
config = {
    "problem_dim": 2,
    "memory": 3,
    "dtype": "single",  # Test with single precision
    "depth": 2,
    "width": 16,
    "activation": "relu",
    "seed": 0
}

@pytest.fixture
def model():
    if config["dtype"] == "double":
        vmin = -1. * np.ones((1,config["problem_dim"],1)).astype("float64")
        vmax = np.ones((1,config["problem_dim"],1)).astype("float64")
    else:
        vmin = -1. * np.ones((1,config["problem_dim"],1)).astype("float32")
        vmax = np.ones((1,config["problem_dim"],1)).astype("float32")
    return resnet(vmin, vmax, config)

def test_resnet_output_shape(model):
    batch_size = 4
    input_dim = config["problem_dim"] * (config["memory"] + 1)

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

def test_resnet_residual_connection(model):
    batch_size = 2
    input_dim = config["problem_dim"] * (config["memory"] + 1)
    # Generate random input with matching dtype
    if config["dtype"] == "double":
        x = np.random.normal(size=(batch_size, input_dim)).astype("float64")
    else:
        x = np.random.normal(size=(batch_size, input_dim)).astype("float32")
    x = from_numpy(x)
    # Forward pass
    output = model(x)

    # Calculate the manual residual addition
    mlp_output = model.mlp(x)
    residual = x[..., -config["problem_dim"]:]
    expected_output = mlp_output + residual

    # Assert the output matches the manually computed residual result
    assert allclose(output, expected_output), \
        "Residual connection in forward pass does not match expected output"

def test_resnet_forward_consistency(model):
    if config["dtype"] == "double":
        x = np.random.normal(size=(4, config["problem_dim"] * (config["memory"] + 1))).astype("float64")
    else:
        x = np.random.normal(size=(4, config["problem_dim"] * (config["memory"] + 1))).astype("float32")
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
        x = np.random.normal(size=(N, config["problem_dim"], config["memory"] + 1)).astype("float64")
    else:
        x = np.random.normal(size=(N, config["problem_dim"], config["memory"] + 1)).astype("float32")
    steps = 5
    # Run predict method
    output = model.predict(x, steps, device)

    # Check if output shape matches expected shape (N, problem_dim, steps+memory+1)
    assert output.shape == (N, config["problem_dim"], steps + config["memory"] + 1), \
        f"Expected output shape {(N, config['problem_dim'], steps + config['memory'] + 1)}, but got {output.shape}"

def test_resnet_seed_reproducibility():
    if config["dtype"] == "double":
        vmin = -1. * np.ones((1,config["problem_dim"],1)).astype("float64")
        vmax = np.ones((1,config["problem_dim"],1)).astype("float64")
    else:
        vmin = -1. * np.ones((1,config["problem_dim"],1)).astype("float32")
        vmax = np.ones((1,config["problem_dim"],1)).astype("float32")
    # Initialize two models with the same seed and config
    model1 = resnet(vmin, vmax, config)
    model2 = resnet(vmin, vmax, config)

    # Random input with matching dtype
    if config["dtype"] == "double":
        x = np.random.normal(size=(4, config["problem_dim"] * (config["memory"] + 1))).astype("float64")
    else:
        x = np.random.normal(size=(4, config["problem_dim"] * (config["memory"] + 1))).astype("float32")
    x = from_numpy(x)

    # Forward pass with both models
    output1 = model1(x)
    output2 = model2(x)

    # Check if outputs from both models are the same
    assert allclose(output1, output2), "Outputs are different for the same seed and input."