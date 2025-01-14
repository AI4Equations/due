import pytest
import numpy as np
import os
import shutil
from due.utils import ode_evaluate, pde1d_evaluate, pde2dirregular_evaluate, pde2dregular_evaluate
from scipy.io import loadmat

def test_ode_evaluate():
    prediction = np.random.rand(10, 2, 4)
    truth = np.random.rand(10, 2, 4)
    save_path = "test_output"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    try:
        ode_evaluate(prediction, truth, save_path)
        
        assert os.path.exists(os.path.join(save_path, "pred.mat"))
        assert os.path.exists(os.path.join(save_path, "rel_err.png"))
        assert os.path.exists(os.path.join(save_path, "rel_err.csv"))
        for i in range(prediction.shape[1]):
            assert os.path.exists(os.path.join(save_path, f"pred_{i}.png"))
        if prediction.shape[1] == 3:
            assert os.path.exists(os.path.join(save_path, "phase.png"))
        
        mat_data = loadmat(os.path.join(save_path, "pred.mat"))
        assert "trajectories" in mat_data
        np.testing.assert_array_equal(mat_data["trajectories"], prediction)
    
    finally:
        shutil.rmtree(save_path)

def test_pde1d_evaluate():
    coordinates = np.linspace(0, 1, 10)
    prediction = np.random.rand(10, 10, 3, 5)
    truth = np.random.rand(10, 10, 3, 5)
    save_path = "test_output"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    try:
        pde1d_evaluate(coordinates, prediction, truth, save_path)
        
        assert os.path.exists(os.path.join(save_path, "pred.mat"))
        assert os.path.exists(os.path.join(save_path, "rel_err.png"))
        assert os.path.exists(os.path.join(save_path, "rel_err.csv"))
        for d in range(3):
            for t in range(5):
                assert os.path.exists(os.path.join(save_path, f"pred_u{d+1}_t{t}.png"))
        
        mat_data = loadmat(os.path.join(save_path, "pred.mat"))
        assert "trajectories" in mat_data
        np.testing.assert_array_equal(mat_data["trajectories"], prediction)
    
    finally:
        shutil.rmtree(save_path)

def test_pde2dregular_evaluate():
    coordinates = np.random.rand(10, 10, 2)
    prediction = np.random.rand(5, 10, 10, 1, 2)
    truth = np.random.rand(5, 10, 10, 1, 2)
    save_path = "test_output"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    try:
        # Test with truth
        pde2dregular_evaluate(coordinates, prediction, truth, save_path)
        assert os.path.exists(os.path.join(save_path, "rel_err.png"))
        assert os.path.exists(os.path.join(save_path, "rel_err.csv"))

        # Test without truth
        pde2dregular_evaluate(coordinates, prediction, None, save_path)
        for d in range(prediction.shape[3]):
            for t in range(prediction.shape[4]):
                assert os.path.exists(os.path.join(save_path, f"pred_u{d+1}_t{t+1}.png"))

    finally:
        shutil.rmtree(save_path)