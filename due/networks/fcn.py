import torch
torch.set_default_dtype(torch.float64)
from .nn import nn
from ..utils import get_activation

class affine(nn):
    def __init__(self, vmin, vmax, config):
        super().__init__()
        self.vmin = torch.from_numpy(vmin)
        self.vmax = torch.from_numpy(vmax)
        self.dtype = config["dtype"]
        self.memory = config["memory"]
        self.output_dim = config["problem_dim"]
        self.input_dim = self.output_dim * (config["memory"] + 1)
        
        self.set_seed(config["seed"])
        if self.dtype == "double":
            self.mDMD = torch.nn.Linear(self.input_dim, self.output_dim).double()
                
        elif self.dtype == "single":
            self.mDMD = torch.nn.Linear(self.input_dim, self.output_dim)
        else:
            print("self.dtype error. The self.dtype must be either single or double.")
            exit()
    def forward(self, x):
        return self.mDMD(x)

    def predict(self, x, steps, device):
        """
        This function is NOT used for training. It is for testing and predicting trajectories given initial states which are not seen during training.
        
        x : unnormalized intial conditions. Numpy array (N, output_dim, memory+1)
        output: unnormalized prediction at the future steps. Numpy array (N, output_dim, steps+1)
        """
        
        self.to(device)
        assert x.shape[1] == self.output_dim
        assert x.shape[2] == self.memory + 1

        xx    = torch.from_numpy(x)
        xx    = 2 * (xx - 0.5*(self.vmax+self.vmin) ) / (self.vmax-self.vmin)
        xx    = xx.to(device)
        
        yy  = torch.zeros(xx.shape[0], self.output_dim, steps+self.memory+1, device=device, dtype=xx.dtype)#torch.zeros_like(y).to(self.device)
        yy[...,:self.memory+1] = xx
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                yy[..., self.memory+1+t] = self.forward(yy[..., t:self.memory+1+t].permute(0,2,1).reshape(-1,self.input_dim))

        yy = yy.cpu()        
        yy = yy * 0.5*(self.vmax-self.vmin) + 0.5*(self.vmax+self.vmin)
        
        return yy.numpy()

class mlp(nn):
    """Fully-connected neural network."""

    def __init__(self, config):
        super().__init__()

        self.dtype = config["dtype"]
        self.output_dim = config["problem_dim"]
        self.memory = config["memory"]
        self.input_dim = self.output_dim * (config["memory"] + 1)
        self.depth = config["depth"]
        self.width = config["width"]
        self.activation = get_activation(config["activation"])
        ################        
        
        self.set_seed(config["seed"])
        self.layers = torch.nn.ModuleList()
        if self.dtype == "double":
            for i in range(self.depth):
                if i==0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.width).double())
                else:
                    self.layers.append(torch.nn.Linear(self.width, self.width).double())
            self.layers.append(torch.nn.Linear(self.width, self.output_dim).double())
                
        elif self.dtype == "single":
            for i in range(self.depth):
                if i==0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.width))
                else:
                    self.layers.append(torch.nn.Linear(self.width, self.width))
            self.layers.append(torch.nn.Linear(self.width, self.output_dim))
        else:
            print("self.dtype error. The self.dtype must be either single or double.")
            exit()
        
    def forward(self, x):
        for i,l in enumerate(self.layers[:-1]):
            x = l(x)
            x = self.activation(x)

        x = self.layers[-1](x)
        return x
        
class resnet(affine):

    def __init__(self, vmin, vmax, config):
        super().__init__(vmin, vmax, config)
        
        self.mlp = mlp(config)
    
    def forward(self, x):
        return self.mlp(x) + x[...,-self.output_dim:]

class gresnet(affine):
    def __init__(self, prior, vmin, vmax, config):
        super().__init__(vmin, vmax, config)
        
        self.prior = prior
        for param in self.prior.parameters():
            param.requires_grad = False
        
        self.mlp = mlp(config)
    
    def forward(self, x):
        return self.prior(x) + self.mlp(x)

class osgnet(nn):

    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super().__init__()
        
        self.vmin       = torch.from_numpy(vmin)
        self.vmax       = torch.from_numpy(vmax)
        self.tmin       = tmin#torch.from_numpy(tmin)
        self.tmax       = tmax#torch.from_numpy(tmax)
        self.dtype      = config["dtype"]
        self.input_dim  = config["problem_dim"]+1
        self.output_dim = config["problem_dim"]
        self.depth      = config["depth"]
        self.width      = config["width"]
        self.activation = get_activation(config["activation"])
        self.multiscale = multiscale
        ################        
        
        self.set_seed(config["seed"])
        
        self.layers = torch.nn.ModuleList()
        if self.dtype == "double":
            for i in range(self.depth):
                if i==0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.width).double())
                else:
                    self.layers.append(torch.nn.Linear(self.width, self.width).double())
            self.layers.append(torch.nn.Linear(self.width, self.output_dim).double())
                
        elif self.dtype == "single":
            for i in range(self.depth):
                if i==0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.width))
                else:
                    self.layers.append(torch.nn.Linear(self.width, self.width))
            self.layers.append(torch.nn.Linear(self.width, self.output_dim))
        else:
            print("self.dtype error. The self.dtype must be either single or double.")
            exit()
        
        self.name = "osgnet"
        
    def forward(self, x):
        try:
            dt = x[:,-1:] * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        except:
            dt = x[:,-1:] * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt
        else:
            pass
            
        xx = x.clone()
        for i,l in enumerate(self.layers[:-1]):
            xx = l(xx)
            xx = self.activation(xx)

        xx = self.layers[-1](xx)
        return x[:,:-1] + xx * dt
        
    def predict(self, x, dt, device):
        """
        This function is NOT used for training. It is for testing and predicting trajectories given initial states which are not seen during training.
        
        x : unnormalized intial conditions. Numpy array (N, output_dim)
        output: unnormalized prediction at the future steps. Numpy array (N, output_dim, steps+1)
        """
        self.to(device)
        assert x.shape[1] == self.output_dim
        steps = dt.shape[1]
        
        dt = torch.from_numpy(dt)
        dt = dt.to(device)
        if self.multiscale:
            dt = torch.log10(dt)
        dt    = 2 * (dt - 0.5*(self.tmax+self.tmin) ) / (self.tmax-self.tmin)
        
        x     = torch.from_numpy(x)
        x     = 2 * (x - 0.5*(self.vmax+self.vmin) ) / (self.vmax-self.vmin)
        x     = x.to(device)

        y  = torch.unsqueeze(x.clone(), -1)
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                xx = torch.cat((y[...,-1], torch.tile(dt[:,t:t+1],[x.shape[0],1])), dim=-1)
                pred = self.forward(xx)
                y    = torch.cat((y,torch.unsqueeze(pred, dim=-1)), dim=-1)

        y = y.cpu()        
        y = y * 0.5*(self.vmax.unsqueeze(-1)-self.vmin.unsqueeze(-1)) + 0.5*(self.vmax.unsqueeze(-1)+self.vmin.unsqueeze(-1))
        
        return y.numpy()
        
class dual_osgnet(osgnet):
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super().__init__(vmin, vmax, tmin, tmax, config, multiscale)
        
        self.osgnet1 = osgnet(vmin, vmax, tmin, tmax, config, multiscale)
        self.osgnet2 = osgnet(vmin, vmax, tmin, tmax, config, multiscale)
        self.gate    = torch.nn.ModuleList()
        if self.osgnet1.dtype == "double":
            self.gate.append(torch.nn.Linear(1, self.osgnet1.width).double())
            self.gate.append(torch.nn.Linear(self.osgnet1.width, 2).double())
        elif self.osgnet1.dtype == "single":
            self.gate.append(torch.nn.Linear(1, self.osgnet1.width))
            self.gate.append(torch.nn.Linear(self.osgnet1.width, 2))
                    
    def forward(self, x):
        
        p  = torch.nn.Softmax(dim=-1)(self.gate[1](self.osgnet1.activation(self.gate[0](x[...,-1:]))))
        y1 = self.osgnet1(x)
        y2 = self.osgnet2(x)
        
        return p[:,:1]*y1 + p[:,1:2]*y2
        
        

        
        
        
        
