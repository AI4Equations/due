import torch
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
from .nn import nn
from ..utils import get_activation
torch.manual_seed(0)
#np.random.seed(0)

class SpectralConv2d(nn):
    """ This is adapted from the implementation of Fourier neural operator. Reference to https://github.com/neuraloperator/neuraloperator/blob/master/fourier_2d_time.py"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn):
    def __init__(self, in_channels, out_channels, mid_channels, activation):
        super(MLP, self).__init__()
        self.mlp1 = torch.nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = torch.nn.Conv2d(mid_channels, out_channels, 1)
        self.activation = activation

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x

class osg_fno2d(nn):
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(osg_fno2d, self).__init__()

        self.vmin       = torch.from_numpy(vmin)
        self.vmax       = torch.from_numpy(vmax)
        self.tmin = tmin
        self.tmax = tmax
        self.input_dim  = config["problem_dim"]+1
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.modes1 = config["modes1"]
        self.modes2 = config["modes2"]
        self.nblocks = config["depth"]
        self.hid_dim  = config["width"]
        self.multiscale = multiscale
        
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        
        self.conv = torch.nn.ModuleList()
        self.mlp  = torch.nn.ModuleList()
        self.w    = torch.nn.ModuleList()
        
        for i in range(self.nblocks):
            self.conv.append(SpectralConv2d(self.hid_dim, self.hid_dim, self.modes1, self.modes2))
            self.mlp.append(MLP(self.hid_dim, self.hid_dim, self.hid_dim, self.activation))
            self.w.append(torch.nn.Conv2d(self.hid_dim, self.hid_dim, 1))
        self.de = MLP(self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation) # output channel is 1: u(x, y)

    def forward(self, x):
        
        x0   = x[...,:-1]
        dt   = x[...,-1:] * 0.5 * (self.tmax-self.tmin) + 0.5 * (self.tmax-self.tmin)
        x = self.en(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = x1 + x2
            x = self.activation(x)

        x = self.de(x)
        x = x.permute(0, 2, 3, 1)
        return x0 + x * dt
        
    def predict(self, x, dt, device):
        """
        This function is NOT used for training. It is for testing and predicting trajectories given initial states which are not seen during training.
        
        x : unnormalized intial conditions. Numpy array (N, H, W, D)
        output: unnormalized prediction at the future steps. Numpy array (N, H, W, steps)
        """
        self.to(device)
        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]
        
        
        dt = torch.from_numpy(dt)
        dt = torch.unsqueeze(dt,1)
        dt = torch.unsqueeze(dt,1)
        dt = torch.tile(dt, [1,x.shape[1],x.shape[2],1])
        dt = dt.to(device)
        if self.multiscale:
            dt = torch.log10(dt)
        dt    = 2 * (dt - 0.5*(self.tmax+self.tmin) ) / (self.tmax-self.tmin)
        
        x     = torch.from_numpy(x)
        x     = 2 * (x - 0.5*(self.vmax[...,0]+self.vmin[...,0]) ) / (self.vmax[...,0]-self.vmin[...,0])
        x     = x.to(device)

        y  = torch.unsqueeze(x.clone(), -1)
        print(x.shape, y.shape, dt.shape)
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                xx = torch.cat((y[...,-1], dt[...,t:t+1]), dim=-1)
                pred = self.forward(xx)
                y    = torch.cat((y,torch.unsqueeze(pred, dim=-1)), dim=-1)

        y = y.cpu()        
        y = y * 0.5*(self.vmax-self.vmin) + 0.5*(self.vmax+self.vmin)
        
        return y.numpy()

