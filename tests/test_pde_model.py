import pytest
import numpy as np
from due.networks.nn import *
from due.networks.transformer import *
from due.models import PDE

@pytest.fixture
def sample_data():
    # Create sample input data for testing
    trainX = np.random.rand(10, 5, 2, 3).astype("float32")  # Example shape (N, L, D, M)
    trainY = np.random.rand(10, 5, 2, 4).astype("float32")  # Example shape (N, L, D, S)
    return trainX, trainY

@pytest.fixture
def config():
    # Sample configuration dictionary for training
    return {
        "seed": 42,
        "device": "cpu",
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": "None",
        "loss": "rel_l2_pde",  # Assuming you have a defined loss function
        "save_path": "./test_save",  # Temporary save path
        "verbose": 1,
    }

@pytest.fixture
def network():
    # Initialize your neural network model
    config_net = {
        "memory": 2,
        "problem_dim": 2,
        "activation": "relu",
        "width": 64,
        "n_head": 8,
        "depth": 1,
        "seed": 42,
        "locality_encoder": 200,
        "locality_decoder": 200,
    }
    mesh1 = np.random.rand(5, 2).astype(np.float32)
    mesh2 = np.random.rand(5, 2).astype(np.float32)
    device = "cpu"
    model = pit(mesh1, mesh2, device, config_net)
    return model.to(device)  # Replace with actual initialization

def test_pde_training(sample_data, config, network):
    trainX, trainY = sample_data
    
    # Create the PDE instance
    pde_model = PDE(trainX, trainY, network, config)

    # Train the model
    pde_model.train()
    
    # Check if the model parameters are saved
    assert os.path.exists(f"{config['save_path']}/model"), "Model was not saved."
    
    # Check if the training history is recorded
    assert pde_model.hist.shape == (config["epochs"], 1), "Training history shape mismatch."

    # Optional: Check if the model can make predictions after training
    pred = pde_model.mynet.predict(trainX, steps=1, device=config["device"])
    assert pred.shape == (10, 5, 2, 4), "Prediction shape mismatch after training."
