import pytest
from torch import from_numpy
import numpy as np
from due.networks.transformer import pit

@pytest.fixture
def setup_pit():
    # Set up a configuration dictionary for the pit class
    config = {
        "memory": 5,
        "problem_dim": 3,
        "activation": "relu",
        "width": 64,
        "n_head": 8,
        "depth": 4,
        "seed": 42,
        "locality_encoder": 2,
        "locality_decoder": 2,
    }
    
    # Example meshes (dummy data)
    mesh1 = np.random.rand(10, 2).astype(np.float32)
    mesh2 = np.random.rand(10, 2).astype(np.float32)

    device = "cpu"

    # Initialize the pit model
    model = pit(mesh1, mesh2, device, config)
    return model.to(device)

def test_forward_pass(setup_pit):
    model = setup_pit
    # Create dummy input data
    input_data = np.random.normal(size=(2, 10, 3, 6)).astype("float32")
    input_data = from_numpy(input_data).to(model.device)  # (batch_size, N, problem_dim, memory+1)
    
    # Perform a forward pass
    output = model(input_data)
    
    # Check the output shape
    assert output.shape == (2, 10, 3), "Output shape mismatch."

def test_predict_method(setup_pit):
    model = setup_pit
    # Create dummy initial conditions
    initial_conditions = np.random.rand(2, 10, 3, 6).astype("float32")  # (N, L, d, memory+1)

    # Test the predict method
    steps = 3
    output = model.predict(initial_conditions, steps, model.device)

    # Check the output shape
    expected_shape = (2, 10, 3, steps + 5 + 1)  # (N, L, d, steps + memory + 1)
    assert output.shape == expected_shape, "Output shape mismatch in predict method."

def test_parameter_count(setup_pit):
    model = setup_pit
    param_count = model.count_params()
    
    # Check that the parameter count is greater than zero
    assert param_count > 0, "The model should have trainable parameters."
