import torch
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
from .nn import *
from ..utils import get_activation
from math import pi

class xavier_mlp(nn): 
    def __init__(self, n_filters0, n_filters1, n_filters2, activation): 
        super(xavier_mlp, self).__init__() 
        self.mlp1 = torch.nn.Linear(n_filters0, n_filters1)
        self.mlp2 = torch.nn.Linear(n_filters1, n_filters2)
        self.activation = activation 
        torch.nn.init.xavier_uniform_(self.mlp1.weight)
        torch.nn.init.xavier_uniform_(self.mlp2.weight)
 
    def forward(self, x): 
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x

class mhsa(nn):
    def __init__(self, n_head, hid_dim, activation):
        super(mhsa, self).__init__()
        self.hid_dim = hid_dim 
        self.n_head = n_head 
        self.v_dim = round(hid_dim / n_head) 
        self.activation = activation

        self.q = torch.nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.q.weight)
        self.k = torch.nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.k.weight) 

    def forward(self, inputs):
        bsize = inputs.shape[0]
        Q = self.q(inputs).view(bsize, -1, self.n_head, self.v_dim).transpose(1, 2) # (b,h,l,d/h)
        K = self.k(inputs).view(bsize, -1, self.n_head, self.v_dim).transpose(1, 2) # (b,h,l,d/h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.v_dim ** 0.5)  # (b,h,l,l)
        att = torch.nn.Softmax(dim=-1)(scores)  # Shape: (b,h,l,l)
        convoluted = torch.einsum('bhli, bid->bhld', att, inputs).transpose(1, 2).contiguous().view(bsize, -1, self.n_head*self.hid_dim)# Shape: (b,l,d)
        return convoluted

class transformer_sa(nn):

    def __init__(self, config):

        super(transformer_sa, self).__init__()

        self.memory     = config["memory"]
        self.input_dim  = config["problem_dim"]*(self.memory+1)
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.hid_dim    = config["width"]
        self.n_head     = config["n_head"]
        self.n_blocks   = config["depth"] - 1
        self.set_seed(config["seed"])
        
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        torch.nn.init.xavier_uniform_(self.en.weight)
        self.sa = mhsa(self.n_head, self.hid_dim, self.activation)

        self.W = torch.nn.ModuleList([torch.nn.Linear(self.n_head*self.hid_dim, self.hid_dim) for _ in range(self.n_blocks)]) 
        for linear in self.W: 
            torch.nn.init.xavier_uniform_(linear.weight) 
        self.SA = torch.nn.ModuleList([mhsa(self.n_head, self.hid_dim, self.activation) for _ in range(self.n_blocks)]) 
 
        self.de= torch.nn.Linear(self.n_head*self.hid_dim, self.output_dim)
        torch.nn.init.xavier_uniform_(self.de.weight)
 
    def forward(self, inputs):
        
        x    = inputs.permute(0,1,3,2).reshape(inputs.shape[0], inputs.shape[1], -1)
        x    = self.en(x)
        x    = self.sa(x)
        x    = self.activation(x)

        for sa, w in zip(self.SA, self.W):
            x = w(x)
            x = sa(x)
            x = self.activation(x) 
        
        x = self.de(x)
        return x + inputs[...,-1]

class mhpa(nn):

    def __init__(self, m_dist, n_head, hid_dim, locality, activation):

        super(mhpa, self).__init__()        
        
        self.dist = m_dist 
        self.locality = locality 
        self.hid_dim = hid_dim 
        self.n_head = n_head 
        self.v_dim = round(hid_dim / n_head) 
        self.activation = activation
 
        self.r = torch.nn.Parameter( torch.rand(n_head, 1, 1) ) 
        self.weight = torch.nn.Parameter( torch.rand(n_head, hid_dim, self.v_dim) )
        torch.nn.init.xavier_uniform_(self.weight) 
 
    def forward(self, inputs): 
        scaled_dist = self.dist * torch.tan(0.25*pi*(1.0+torch.sin(self.r)))
        if self.locality <= 1: 
            mask = torch.quantile(scaled_dist, self.locality, dim=-1, keepdim=True) 
            scaled_dist = torch.where(scaled_dist <= mask, scaled_dist, torch.tensor(torch.finfo(torch.float32).max, device=scaled_dist.device)) 
         
        scaled_dist = -scaled_dist
        att = torch.nn.Softmax(dim=-1)(scaled_dist)  # (n_heads, N, N) 
 
        value = torch.einsum("bnj,hjk->bhnk", inputs, self.weight)  # (batch, n_head, N, v_dim) 
 
        concat = torch.einsum("hnj,bhjd->bhnd", att, value)  # (batch, n_head, N, v_dim) 
        concat = concat.permute(0, 2, 1, 3).contiguous() 
        concat = concat.view(inputs.shape[0], -1, self.hid_dim) 
 
        return self.activation(concat)
       
class pit(pit_fixdt):

    def __init__(self, mesh1, mesh2, device, config):

        super(pit, self).__init__(mesh1, mesh2, device, config)
        
        self.en = torch.nn.Linear(self.input_dim+self.msh_qry.shape[-1], self.hid_dim) 
        torch.nn.init.xavier_uniform_(self.en.weight) 
        self.down = mhpa(self.m_cross.permute(1,0), self.n_head, self.hid_dim, self.en_local, self.activation) 
         
        self.PA = torch.nn.ModuleList([mhpa(self.m_latent, self.n_head, self.hid_dim, 2, self.activation) for _ in range(self.n_blocks)]) 
        self.MLP = torch.nn.ModuleList([xavier_mlp(self.hid_dim, self.hid_dim, self.hid_dim, self.activation) for _ in range(self.n_blocks)]) 
        self.W = torch.nn.ModuleList([torch.nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_blocks)]) 
        for linear in self.W: 
            torch.nn.init.xavier_uniform_(linear.weight) 
 
        self.up = mhpa(self.m_cross, self.n_head, self.hid_dim, self.de_local, self.activation) 
        self.de_layer = xavier_mlp(self.hid_dim, self.hid_dim, self.output_dim, self.activation) 
 
    def forward(self, inputs):
        x    = inputs.permute(0,1,3,2).reshape(inputs.shape[0], inputs.shape[1], -1)
        mesh = self.get_mesh(inputs)
        x    = torch.cat((mesh,x), dim=-1)
        x    = self.activation(self.en(x)) 
        x    = self.down(x) 
 
        for pa, mlp, w in zip(self.PA, self.MLP, self.W): 
            x = mlp(pa(x)) + w(x) 
            x = self.activation(x) 
 
        x = self.up(x) 
        x = self.de_layer(x) 
        return x + inputs[...,-1]
        
