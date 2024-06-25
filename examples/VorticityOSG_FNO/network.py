import torch
from due.networks.nn import *
from due.networks.fno import *

class res_fno(osg_fno2d):

    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):

        super(res_fno, self).__init__(vmin, vmax, tmin, tmax, config, multiscale)
 
    def forward(self, x):
        
        x0   = x[...,:-1]
        # dt   = x[...,-1:] * 0.5 * (self.tmax-self.tmin) + 0.5 * (self.tmax-self.tmin)
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
        return x0 + x
