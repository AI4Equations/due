import pytest
import numpy as np
import torch
from due.networks.fcn import dual_osgnet

# Configuration dictionary with example values
config = {
    "problem_dim": 2,
    "memory": 0,
    "dtype": "single",  # Test with single precision
    "depth": 2,
    "width": 16,
    "activation": "relu",
    "seed": 0
}

@pytest.fixture
def dual_model():
    if config["dtype"] == "double":
        vmin = -1. * np.ones((1, config["problem_dim"])).astype("float64")
        vmax = np.ones((1, config["problem_dim"])).astype("float64")
        tmin = 0.5
        tmax = 1.5
    else:
        vmin = -1. * np.ones((1, config["problem_dim"])).astype("float32")
        vmax = np.ones((1, config["problem_dim"])).astype("float32")
        tmin = 0.5
        tmax = 1.5
    return dual_osgnet(vmin, vmax, tmin, tmax, config)

def test_dual_osgnet_output_shape(dual_model):
    batch_size = 4
    input_dim = config["problem_dim"] + 1

    # Generate random input with matching dtype
    if config["dtype"] == "double":
        x = np.random.normal(size=(batch_size, input_dim)).astype("float64")
    else:
        x = np.random.normal(size=(batch_size, input_dim)).astype("float32")
    x = torch.from_numpy(x)
    # Forward pass
    output = dual_model(x)

    # Check if output shape matches the expected shape (batch_size, problem_dim)
    assert output.shape == (batch_size, config["problem_dim"]), \
        f"Expected output shape {(batch_size, config['problem_dim'])}, but got {output.shape}"

def test_dual_osgnet_forward_consistency(dual_model):
    if config["dtype"] == "double":
        x = np.random.normal(size=(4, (config["problem_dim"] + 1))).astype("float64")
    else:
        x = np.random.normal(size=(4, (config["problem_dim"] + 1))).astype("float32")
    x = torch.from_numpy(x)
    # First forward pass
    output1 = dual_model(x)
    # Second forward pass
    output2 = dual_model(x)
    # Ensure consistency of outputs for the same input
    assert torch.allclose(output1, output2), \
        "Forward pass is inconsistent for the same input"

def test_dual_osgnet_gate_functionality(dual_model):
    batch_size = 4
    input_dim = config["problem_dim"] + 1

    # Generate random input with matching dtype
    if config["dtype"] == "double":
        x = np.random.normal(size=(batch_size, input_dim)).astype("float64")
    else:
        x = np.random.normal(size=(batch_size, input_dim)).astype("float32")
    x = torch.from_numpy(x)
    # Forward pass
    output = dual_model(x)

    # Check if gate outputs probabilities that sum to 1
    gate_output = dual_model.gate[1](dual_model.osgnet1.activation(dual_model.gate[0](x[..., -1:])))
    p = torch.nn.Softmax(dim=-1)(gate_output)
    assert torch.allclose(p.sum(dim=-1), torch.ones(batch_size)), \
        "Gate output probabilities do not sum to 1"

def test_dual_osgnet_seed_reproducibility():
    if config["dtype"] == "double":
        vmin = -1. * np.ones((1, config["problem_dim"])).astype("float64")
        vmax = np.ones((1, config["problem_dim"])).astype("float64")
        tmin = 0.5
        tmax = 1.5
    else:
        vmin = -1. * np.ones((1, config["problem_dim"])).astype("float32")
        vmax = np.ones((1, config["problem_dim"])).astype("float32")
        tmin = 0.5
        tmax = 1.5
    # Initialize two models with the same seed and config
    model1 = dual_osgnet(vmin, vmax, tmin, tmax, config)
    model2 = dual_osgnet(vmin, vmax, tmin, tmax, config)

    # Random input with matching dtype
    if config["dtype"] == "double":
        x = np.random.normal(size=(4, (config["problem_dim"] + 1))).astype("float64")
    else:
        x = np.random.normal(size=(4, (config["problem_dim"] + 1))).astype("float32")
    x = torch.from_numpy(x)

    # Forward pass with both models
    output1 = model1(x)
    output2 = model2(x)

    # Check if outputs from both models are the same
    assert torch.allclose(output1, output2), "Outputs are different for the same seed and input."