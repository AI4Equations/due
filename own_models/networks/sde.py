import torch
from due.networks.fcn import affine
from due.utils import get_activation


class SDEResNet(affine):
    """
    Conditional generative network G_theta(x, z) approximating the stochastic
    flow map F_dt(x, omega) of Eq. (2.3)-(2.4) in the paper.

    Unlike due.networks.fcn.resnet (which maps x -> x + mlp(x)), this network's
    input is the concatenation of the physical state x (with its memory
    embedding, if any) and a standard normal draw z, flattened along the last
    axis: [x, z] with shape (..., input_dim + output_dim). The residual
    connection only adds back the *physical state* (the most recent memory
    step), never the noise, matching the paper's supervised flow-map training
    (Section 3.3): G_theta(x, z) = mlp([x, z]) + x.

    Built directly on due.networks.fcn.affine so it inherits vmin/vmax
    denormalization, the seeding convention, and count_params()/load_params()
    for free, and can be constructed and trained exactly like any other DUE
    network: SDEResNet(vmin, vmax, config), then due.models.ODE(...).
    """

    def __init__(self, vmin, vmax, config):
        super().__init__(vmin, vmax, config)
        # affine.__init__ already set: self.dtype, self.memory, self.output_dim,
        # self.input_dim = output_dim * (memory + 1)  (the physical-state condition x)

        self.depth = config["depth"]
        self.width = config["width"]
        self.activation = get_activation(config["activation"])

        # The network additionally consumes a noise vector z of size output_dim.
        mlp_input_dim = self.input_dim + self.output_dim

        self.layers = torch.nn.ModuleList()
        if self.dtype == "double":
            for i in range(self.depth):
                in_dim = mlp_input_dim if i == 0 else self.width
                self.layers.append(torch.nn.Linear(in_dim, self.width).double())
            self.layers.append(torch.nn.Linear(self.width, self.output_dim).double())
        elif self.dtype == "single":
            for i in range(self.depth):
                in_dim = mlp_input_dim if i == 0 else self.width
                self.layers.append(torch.nn.Linear(in_dim, self.width))
            self.layers.append(torch.nn.Linear(self.width, self.output_dim))
        else:
            print("self.dtype error. The self.dtype must be either single or double.")
            exit()

    def forward(self, xz):
        """
        xz: (..., input_dim + output_dim), the flattened concatenation of the
        physical-state condition x and the noise draw z.
        """
        xz = xz.to(self.layers[0].weight.dtype)
        x_last = xz[..., self.input_dim - self.output_dim:self.input_dim]

        h = xz
        for l in self.layers[:-1]:
            h = self.activation(l(h))
        h = self.layers[-1](h)

        return h + x_last

    def predict(self, x, steps, device):
        """
        Autoregressive long-term trajectory generation (not used for training).

        x : unnormalized initial conditions. Numpy array (N, output_dim, memory+1)
        output: unnormalized generated trajectories. Numpy array (N, output_dim, steps+memory+1)

        At every step, a fresh z ~ N(0, I) is drawn and concatenated with the
        current physical state before calling forward(), exactly as required
        by the conditional generative model G_theta(x, z) (Eq. 2.4): the
        stochastic flow map has a genuinely random forward evolution, so the
        noise cannot be fixed or reused across steps or samples.
        """
        self.to(device)
        assert x.shape[1] == self.output_dim
        assert x.shape[2] == self.memory + 1

        xx = torch.from_numpy(x)
        xx = 2 * (xx - 0.5 * (self.vmax + self.vmin)) / (self.vmax - self.vmin)
        xx = xx.to(device).to(self.layers[0].weight.dtype)

        yy = torch.zeros(xx.shape[0], self.output_dim, steps + self.memory + 1, device=device, dtype=xx.dtype)
        yy[..., :self.memory + 1] = xx
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                x_in = yy[..., t:self.memory + 1 + t].permute(0, 2, 1).reshape(-1, self.input_dim)
                z = torch.randn(x_in.shape[0], self.output_dim, device=device, dtype=x_in.dtype)
                xz = torch.cat([x_in, z], dim=-1)
                yy[..., self.memory + 1 + t] = self.forward(xz)

        yy = yy.cpu()
        yy = yy * 0.5 * (self.vmax - self.vmin) + 0.5 * (self.vmax + self.vmin)

        return yy.numpy()
