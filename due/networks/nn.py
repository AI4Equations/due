import torch
import os
from ..utils import get_activation

class nn(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()

    def count_params(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)
        
    def load_params(self, save_path):
        """Evaluate the number of trainable parameters for the NN."""
        return torch.load(save_path)
        
    def set_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
       
class pit_fixdt(nn):
    
    """Base class for Position-induced Transformers.
    
    Args:
        mesh1 (ndarray): The first mesh array of shape (L1, d).
        mesh2 (ndarray): The second mesh array of shape (L2, d).
        device (str): The device to run the computations on.
        config (dict): Configuration parameters for the class.
        
    Attributes:
        device (str): The device to run the computations on.
        msh_qry (ndarray): The mesh for training/testing data, array of shape (L_qry, d).
        msh_ltt (ndarray): The pre-fixed latent mesh for pit, array of shape (L_ltt, d).
        m_cross (ndarray): Pairwise distance between msh_qry and msh_ltt.
        m_latent (ndarray): Pairwise distance between msh_ltt and msh_ltt.
        npoints (int): Number of points in msh_qry.
        memory (int): Memory step.
        input_dim (int): Input channels.
        output_dim (int): Output channels.
        activation (function): Activation function.
        hid_dim (int): Hidden channels/network width/lifting dimension.
        n_head (int): Number of attention heads.
        n_blocks (int): Number of attention blocks.
        en_local (float): quantile for local attention in the encoder.
        de_local (float): quantile for local attention in the decoder.
    """
    
    def __init__(self, mesh1, mesh2, device, config):
        super(pit_fixdt, self).__init__()
        
        self.device     = device
        self.msh_qry  = mesh1 # (N,2)
        self.msh_ltt  = mesh2.astype(mesh1.dtype)
        self.m_cross  = self.pairwise_dist(self.msh_qry, self.msh_ltt, self.device)
        self.m_latent = self.pairwise_dist(self.msh_ltt, self.msh_ltt, self.device)
        self.npoints  = self.msh_qry.shape[0]
        
        self.memory     = config["memory"]
        self.input_dim  = config["problem_dim"]*(self.memory+1)
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.hid_dim    = config["width"]
        self.n_head     = config["n_head"]
        self.n_blocks   = config["depth"]
        self.en_local   = config['locality_encoder']
        self.de_local   = config['locality_decoder']
        self.set_seed(config["seed"])
        
    def get_mesh(self, inputs):
        mesh = torch.from_numpy(self.msh_qry)
        mesh = mesh.to(self.device)
        return torch.tile(torch.unsqueeze(mesh, dim=0), [inputs.shape[0], 1, 1])
     
    def predict(self, x, steps, device):
        """
        This function is NOT used for training. It is for testing, producing trajectories given initial states which are not seen during training.
        
        x : intial conditions. Numpy array (N, L, d, memory)
        output: unnormalized prediction at the future steps. Numpy array (N, L, d, steps)
        """

        xx    = torch.from_numpy(x)
        xx    = xx.to(device)
        
        yy  = torch.zeros(*xx.shape[:-1], steps+self.memory+1, device=device)
        yy[...,:self.memory+1] = xx
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                yy[..., self.memory+t+1] = self.forward(yy[..., t:self.memory+t+1])
        
        return yy.cpu().numpy()
        
    def pairwise_dist(self, mesh1, mesh2, device):
        try:
            mesh1 = torch.from_numpy(mesh1)
            mesh2 = torch.from_numpy(mesh2)
        except:
            pass
        mesh1 = mesh1.to(device)
        mesh2 = mesh2.to(device)
        dist  = torch.cdist(mesh1, mesh2, p=2)
        dist2 = dist**2
        dist2 = dist2/torch.max(dist2)
        return dist2
        
