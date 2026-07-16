import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.io import savemat
from yaml import safe_load
from pathlib import Path

def read_config(config_path):

    config = safe_load(Path(config_path).read_text())
    
    conf_data  = config["data"]
    conf_data["seed"] = config["seed"]
    conf_data["dtype"] = config["dtype"]
    
    conf_net   = config["network"]
    try:
        conf_net["memory"] = conf_data["memory"]
    except:
        pass
    
    conf_net["seed"] = config["seed"]
    conf_net["dtype"] = config["dtype"]
    
    conf_train = config["training"]
    conf_train["seed"] = config["seed"]
    conf_train["dtype"] = config["dtype"]
    conf_net["device"] = conf_train["device"]
    try:
        if (conf_data["problem_type"]=="1d_irregular") or (conf_data["problem_type"]=="1d_regular"):
            conf_net["problem_dim"] = 2*conf_train["modes"] + 1
        elif (conf_data["problem_type"]=="2d_irregular") or (conf_data["problem_type"]=="2d_regular"):
            conf_net["problem_dim"] = (2*conf_train["modes"] + 1)**2
        else:
            conf_net["problem_dim"] = conf_data["problem_dim"]
    except:
        conf_net["problem_dim"] = conf_data["problem_dim"]
    
    return conf_data, conf_net, conf_train

def read_csv(path, dtype):
    """
    use comma as separator
    """
    data = np.genfromtxt(path, delimiter=',')
    if dtype=="single":
        return data.astype("float32")
    else:
        return data.astype("float64")

def pde2dirregular_evaluate(coordinates, elements, prediction, truth, save_path):
    """
    evaluate the relative l2 error of prediction and references, for two-dimensional PDE data collected on irregular meshes
    If ground truth is None, do plot only.
    """
    assert len(prediction.shape) == 4
    savemat(save_path+"/pred.mat", mdict={"trajectories":prediction})
    N = prediction.shape[0]
    L = prediction.shape[1]
    D = prediction.shape[2]
    T = prediction.shape[3]
    triangulation = tri.Triangulation(coordinates[:,0], coordinates[:,1], elements)
    
    if truth is None:
        vmax = np.max(prediction[-1,...], axis=(0,2))#(D,)
        vmin = np.min(prediction[-1,...], axis=(0,2))#(D,)
        for d in range(D):
            for t in range(T):
                plt.figure(figsize=(9,9), dpi=300)
                plt.axes([0,0,1,1])
                plt.imshow(prediction[-1,:,:,d,t], vmax=vmax[d], vmin=vmin[d], interpolation='spline16', cmap='jet')
                plt.axis('off')
                plt.axis('equal')
                plt.savefig(save_path+"/pred_u{}_t{}.png".format(d+1,t+1))
                plt.close()
                
    else:
        assert prediction.shape == truth.shape
        print("Relative error is:", np.mean(np.linalg.norm((prediction[...,1:]-truth[...,1:]).transpose(0,3,1,2).reshape(N,-1,D), ord=2, axis=1) / np.linalg.norm(truth[...,1:].transpose(0,3,1,2).reshape(N,-1,D), ord=2, axis=1)))
        l2_rel_err = np.linalg.norm((prediction-truth), ord=2, axis=1) / np.linalg.norm(truth, ord=2, axis=1) # (N,D,T)
        l2_rel_err = np.mean(l2_rel_err, axis=0) # (D,T)
        plt.figure(figsize=(9,9), dpi=300)
        for d in range(D):
            plt.plot(np.arange(T), l2_rel_err[d,:], label=r"$u_{{{}}}$".format(d+1))
        plt.legend()
        plt.savefig(save_path+"/rel_err.png")
        plt.close()
        np.savetxt(save_path+"/rel_err.csv", l2_rel_err.T)
        
        vmax = np.max(prediction[-1,...], axis=(0,2))
        vmin = np.min(prediction[-1,...], axis=(0,2))
        abs_err = np.abs(prediction-truth)[-1,...]
        emax = np.max(abs_err, axis=(0,2))
        emin = np.min(abs_err, axis=(0,2))
        for d in range(D):
            print("Plot variable {}.".format(d+1))
            for t in range(T):
                print("    Plot time {}.".format(t))
                plt.figure(figsize=(8,4),dpi=300)
                plt.axes([0,0,1,1])
                plt.tricontourf(triangulation, truth[-1,:,d,t], vmax=vmax[d], vmin=vmin[d], levels=512, cmap='jet')
                plt.axis('off')
                plt.axis('equal')
                plt.savefig(save_path+'/true_variable{}_time{}.png'.format(d+1,t))
                plt.close()

                plt.figure(figsize=(8,4),dpi=300)
                plt.axes([0,0,1,1])
                plt.tricontourf(triangulation, prediction[-1,:,d,t], vmax=vmax[d], vmin=vmin[d], levels=512, cmap='jet')
                plt.axis('off')
                plt.axis('equal')
                plt.savefig(save_path+'/pred_variable{}_time{}.png'.format(d+1,t))
                plt.close()

                plt.figure(figsize=(8,4),dpi=300)
                plt.axes([0,0,1,1])
                plt.tricontourf(triangulation, abs_err[:,d,t], vmax=emax[d], vmin=emin[d], levels=512, cmap='jet')
                plt.axis('off')
                plt.axis('equal')
                plt.savefig(save_path+'/err_variable{}_time{}.png'.format(d+1,t))
                plt.close()

def pde2dregular_evaluate(coordinates, prediction, truth, save_path):
    """
    evaluate the relative l2 error of prediction and references, for two-dimensional PDE data collected on regular grids
    If ground truth is None, do plot only.
    """

    assert len(prediction.shape) == 5
    savemat(save_path+"/pred.mat", mdict={"trajectories":prediction})
    N = prediction.shape[0]
    H = prediction.shape[1]
    W = prediction.shape[2]
    D = prediction.shape[3]
    T = prediction.shape[4]
    
    if truth is None:
        vmax = np.max(prediction[-1,...], axis=(0,1,3))
        vmin = np.min(prediction[-1,...], axis=(0,1,3))
        for d in range(D):
            for t in range(T):
                plt.figure(figsize=(9,9), dpi=300)
                plt.axes([0,0,1,1])
                plt.imshow(prediction[-1,:,:,d,t], vmax=vmax[d], vmin=vmin[d], interpolation='spline16', cmap='jet')
                plt.axis('off')
                plt.axis('equal')
                plt.savefig(save_path+"/pred_u{}_t{}.png".format(d+1,t+1))
                plt.close()
                
    else:
        assert prediction.shape == truth.shape
        print("Relative error is:", np.mean(np.linalg.norm((prediction-truth).transpose(0,4,1,2,3).reshape(N,-1,D), ord=2, axis=1) / np.linalg.norm(truth.transpose(0,4,1,2,3).reshape(N,-1,D), ord=2, axis=1)))
        l2_rel_err = np.linalg.norm((prediction-truth).reshape(N,H*W,D,T), ord=2, axis=1) / np.linalg.norm(truth.reshape(N,H*W,D,T), ord=2, axis=1) # (N,D,T)
        l2_rel_err = np.mean(l2_rel_err, axis=0) # (D,T)
        plt.figure(figsize=(9,9), dpi=300)
        for d in range(D):
            plt.plot(np.arange(T), l2_rel_err[d,:], label=r"$u_{{{}}}$".format(d+1))
        plt.legend()
        plt.savefig(save_path+"/rel_err.png")
        plt.close()
        np.savetxt(save_path+"/rel_err.csv", l2_rel_err.T)
        
def pde1d_evaluate(coordinates, prediction, truth, save_path):
    """
    evaluate the relative l2 error of prediction and references, for one-dimensional PDE data collected on either regular grids or irregular meshes
    If ground truth is None, do plot only.
    """
    
    assert len(prediction.shape) < 5
    savemat(save_path+"/pred.mat", mdict={"trajectories":prediction})
    N = prediction.shape[0]
    L = prediction.shape[1]
    D = prediction.shape[2]
    T = prediction.shape[3] 
    if truth is None:
        for d in range(D):
            for t in range(T):
                plt.figure(figsize=(9,9), dpi=300)
                plt.plot(coordinates, prediction[-1,:,d,t], linestyle="dashed", color="blue", label="pred")
                plt.legend()
                plt.savefig(save_path+"/pred_u{}_t{}.png".format(d+1,t))
                plt.close()
    else:
        assert prediction.shape == truth.shape
        l2_rel_err = np.linalg.norm(prediction-truth, ord=2, axis=1)# / np.linalg.norm(truth, ord=2, axis=1) # (N,D,T)
        l2_rel_err = np.mean(l2_rel_err, axis=0) # (D,T)
        plt.figure(figsize=(9,9), dpi=300)
        for d in range(D):
            plt.plot(np.arange(T), l2_rel_err[d,:], label=r"$u_{{{}}}$".format(d+1))
        plt.legend()
        plt.savefig(save_path+"/rel_err.png")
        plt.close()
        np.savetxt(save_path+"/rel_err.csv", l2_rel_err.T)
        
        for d in range(D):
            for t in range(T):
                plt.figure(figsize=(9,9), dpi=300)
                plt.plot(coordinates, truth[-1,:,d,t], color="red", label="true")
                plt.plot(coordinates, prediction[-1,:,d,t], linestyle="dashed", color="blue", label="pred")
                plt.legend()
                plt.savefig(save_path+"/pred_u{}_t{}.png".format(d+1,t))
                plt.close()
                
def ode_evaluate(prediction, truth, save_path):
    """
    evaluate a batch of predicted trajectories, against the ground truth.
    If ground truth is None, do plot only.
    """
    
    savemat(save_path+"/pred.mat", mdict={"trajectories":prediction})
    dim = prediction.shape[1]
    if truth is None:
        steps = prediction.shape[2]
        for i in range(dim):
            plt.figure(figsize=(9,9), dpi=300)
            plt.plot(np.arange(steps), prediction[-1,i,:], linestyle="dashed", color="blue", label="pred")
            plt.legend()
            plt.savefig(save_path+"/pred_{}.png".format(i))
            plt.close()
    
        if dim==2:
            print("Two state variables. PLotting the 2D phase plot.")
            plt.figure(figsize=(9,9), dpi=300)
            plt.plot(prediction[-1,0,:], prediction[-1,1,:], linestyle="dashed", color="blue", label="pred")
            plt.legend()
            plt.savefig(save_path+"/phase.png")
            plt.close()
        
        elif dim==3:
            print("Three state variables. PLotting the 3D phase plot.")
            fig = plt.figure(figsize=(9,9), dpi=300)
            ax = plt.axes(projection='3d')
            ax.plot3D(prediction[-1,0,:], prediction[-1,1,:], prediction[-1,2,:], linestyle="dashed", color="blue", label="pred")
            plt.legend()
            plt.savefig(save_path+"/phase.png")
            plt.close()
            
    else:
        assert prediction.shape == truth.shape
        steps = truth.shape[2]
        
        rel_err = np.mean(np.abs(prediction-truth), axis=0) # (dim, steps)
        plt.figure(figsize=(9,9), dpi=300)
        for i in range(dim):
            plt.plot(np.arange(steps), rel_err[i,:], label=r"$u_{{{}}}$".format(i+1))
        plt.legend()
        plt.savefig(save_path+"/rel_err.png")
        plt.close()
        np.savetxt(save_path+"/rel_err.csv", rel_err.T)
        
        for i in range(dim):
            plt.figure(figsize=(9,9), dpi=300)
            plt.plot(np.arange(steps), truth[-1,i,:], color="red", label="true")
            plt.plot(np.arange(steps), prediction[-1,i,:], linestyle="dashed", color="blue", label="pred")
            plt.legend()
            plt.savefig(save_path+"/pred_{}.png".format(i))
            plt.close()
        
        if dim==2:
            print("Two state variables. PLotting the 2D phase plot.")
            plt.figure(figsize=(9,9), dpi=300)
            plt.plot(truth[-1,0,:], truth[-1,1,:], color="red", label="true")
            plt.plot(prediction[-1,0,:], prediction[-1,1,:], linestyle="dashed", color="blue", label="pred")
            plt.legend()
            plt.savefig(save_path+"/phase.png")
            plt.close()
        
        elif dim==3:
            print("Three state variables. PLotting the 3D phase plot.")
            fig = plt.figure(figsize=(9,9), dpi=300)
            ax = plt.axes(projection='3d')
            ax.plot3D(truth[-1,0,:], truth[-1,1,:], truth[-1,2,:], color="red", label="true")
            ax.plot3D(prediction[-1,0,:], prediction[-1,1,:], prediction[-1,2,:], linestyle="dashed", color="blue", label="pred")
            plt.legend()
            plt.savefig(save_path+"/phase.png")
            plt.close()

class generalized_fourier_projection1d():
    """
    Used for learning one-domensional PDEs in modal spaces. 
    It provides a forward funtion to compute the modal coefficients of a batch of PDE data, and a backward function to recover the PDE solutions in the physical space from a batch of modal coefficients.
    Currently, it supports the trignometric basis functions.
    """    
    def __init__(self, coords, config_train):
        self.coords = coords
        self.modes  = config_train["modes"]
        del config_train
        self.sinx = np.sin(self.coords @ (np.arange(1,self.modes+1)[np.newaxis,:])) #(L, modes)
        self.cosx = np.cos(self.coords @ (np.arange(1,self.modes+1)[np.newaxis,:]))
        self.A    = np.hstack((np.ones_like(self.coords), self.sinx, self.cosx)) # (L, 2*modes+1)

    def forward(self, trainX, trainY, training):
        """
        Input data shape: (batch_size,L,D,T)
        Output Fourier coefficients shape: (batch_size,2*modes+1,D,T)
        """
        
        memory = trainX.shape[-1]-1
        steps  = trainY.shape[-1]
        data = np.concatenate((trainX, trainY), axis=-1)
        self.N = data.shape[0]
        self.L = data.shape[1]
        self.D = data.shape[2]
        self.T = data.shape[3]
        if training == True:
            assert self.T == memory + steps + 1
        else:
            pass
        data = data.transpose(1,0,2,3) #(L,N,D,T)
        Coeff = np.zeros((2*self.modes+1, self.N, self.D, self.T))
        for d in range(self.D):
            for t in range(self.T):
                Coeff[...,d,t] = np.linalg.lstsq(self.A, data[:,:,d,t], rcond=None)[0]
                
        Coeff = Coeff.transpose(1,0,2,3) #(N,2*modes+1,D,T)
        if training == True:
            #normalization
            Coeff, self.vmin, self.vmax = self.normalize(Coeff)
            print("Training data is normalized")
            trainX = Coeff[...,:memory+1].transpose(0,3,2,1).reshape(self.N,-1) # (N, (2*modes+1)*D*M)
            trainY = Coeff[...,memory+1:].transpose(0,2,1,3).reshape(self.N,-1,steps) # (N,(2*modes+1)D,S)
            print(trainX.shape, trainY.shape)
            return trainX, trainY, self.vmin, self.vmax
        else:
            
#            Coeff = Coeff.transpose(0,2,1,3).reshape(self.N,-1,self.T)
            return Coeff[...,:memory+1].transpose(0,2,1,3).reshape(self.N,-1,memory+1), Coeff[...,memory+1:].transpose(0,2,1,3).reshape(self.N,-1,steps)

    def backward(self, pred_modal):
        """
        Input Fourier coefficients shape: (batch_size,(2*modes+1)*D,T), a numpy array
        Output data shape: (batch_size,L,D,T)
        Used for testing and prediction
        """
        pred_modal = pred_modal.reshape(self.N, self.D, 2*self.modes+1, self.T).transpose(0,2,1,3) #(N,2*modes+1,D,T)
        pred       = np.einsum("lk,nkdt->nldt", self.A, pred_modal)
        
        return pred
       
    def normalize(self, data):
    
        axes = tuple(np.delete(np.arange(len(data.shape)),[-2]))
        vmax = np.amax(data, axis=axes, keepdims=True)
        vmin = np.amin(data, axis=axes, keepdims=True)
        target = 2*(data-0.5*(vmax+vmin))/(vmax-vmin)
        return target, vmin[0,...], vmax[0,...]        
        
def rel_l1_norm(true, pred):
    
    bsize = true.shape[0]
    rel_error  = torch.norm(true.reshape(bsize,-1)-pred.reshape(bsize,-1), p=1, dim=1) / torch.norm(true.reshape(bsize,-1), p=1, dim=1)#(bsize,)
    return torch.mean(rel_error)

def rel_l2_norm(true, pred):
    
    bsize = true.shape[0]
    rel_error  = torch.norm(true.reshape(bsize,-1)-pred.reshape(bsize,-1), p=2, dim=1) / torch.norm(true.reshape(bsize,-1), p=2, dim=1)#(bsize,)
    return torch.mean(rel_error)
    
def rel_l2_norm_pde(true, pred):
    """
    true, pred: (N,L,D,T)
    """
    true = true.reshape(true.shape[0], -1, true.shape[-2], true.shape[-1])
    pred = pred.reshape(pred.shape[0], -1, pred.shape[-2], pred.shape[-1])
    rel_error  = torch.norm(true-pred, p=2, dim=1) / torch.norm(true, p=2, dim=1)#(N,D,T)
    return torch.mean(rel_error)

def rel_l1_norm_pde(true, pred):
    """
    true, pred: (N,L,D,T)
    """
    rel_error  = torch.norm(true-pred, p=1, dim=1) / torch.norm(true, p=1, dim=1)#(N,D,T)
    return torch.mean(rel_error)
    
def get_activation(name):

    if name in ['tanh', 'Tanh']:
        return torch.nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return torch.nn.ReLU(inplace=True)
    elif name in ['leaky_relu', 'LeakyReLU']:
        return torch.nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return torch.nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return torch.nn.Softplus()
    elif name in ['gelu', 'Gelu']:
        return torch.nn.functional.gelu
        
    else:
        raise ValueError(f'unknown or unsupported activation function: {name}')
        
def get_optimizer(name, model, lr):

    if name in ['adam', 'Adam', 'ADAM']:
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name in ['nadam', 'NAdam', 'NADAM']:
        return torch.optim.NAdam(model.parameters(), lr=lr)
    elif name in ['adamw', 'AdamW', 'ADAMW']:
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif name in ['SGD', 'sgd', 'Sgd']:
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f'unknown or unsupported optimizer: {name}')
        
def get_schedule(optimizer, name, epochs, batch_size, ntrain):

    if name in ['cyclic_cosine', 'Cyclic_cosine', 'Cyclic_Cosine']:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=(epochs//5)*(ntrain//batch_size))
    
    elif name in ['cosine', 'Cosine']:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * (ntrain//batch_size))
    elif name in ['one_cycle', 'One_Cycle', 'OneCycle']:
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=epochs * (ntrain//batch_size))
        
    elif name in ['none', 'None']:
        return None
        
    else:
        raise ValueError(f'unknown or unsupported learning schedule: {name}')
        
def get_loss(name):
    if name in ['mse', 'Mse', 'MSE']:
        return torch.nn.MSELoss(reduction="mean")
    elif name in ['mae', 'Mae', 'MAE']:
        return torch.nn.L1Loss(reduction="mean")
    elif name in ['rel_l2', 'Rel_l2', 'Rel_L2']:
        return rel_l2_norm
    elif name in ['rel_l2_pde', 'Rel_l2_pde', 'Rel_L2_pde']:
        return rel_l2_norm_pde
    elif name in ['rel_l1_pde', 'Rel_l1_pde', 'Rel_L1_pde']:
        return rel_l1_norm_pde
    elif name in ['rel_l1', 'Rel_l1', 'Rel_L1']:
        return rel_l1_norm
    else:
        raise ValueError(f'unknown or unsupported loss function: {name}')


def normalize_state(x, vmin, vmax):
    """
    Applies [-1,1] min-max normalization against already-known bounds,
    rather than computing them from x. Used by
    due.models.sde_diffusion.generate_labeled_data to normalize the
    diffusion engine's raw-unit outputs right before they reach the network.

    x: (..., dim) tensor or ndarray, raw physical units.
    vmin, vmax: (1, dim, 1) ndarrays, as returned by sde_dataset.load().

    Returns the same type as x (tensor in, tensor out; ndarray in, ndarray out).
    """
    vmin_flat = np.asarray(vmin).reshape(-1)
    vmax_flat = np.asarray(vmax).reshape(-1)
    range_flat = vmax_flat - vmin_flat
    range_flat = np.where(range_flat == 0, 1.0, range_flat)
    center = 0.5 * (vmax_flat + vmin_flat)

    if isinstance(x, torch.Tensor):
        center_t = torch.as_tensor(center, dtype=x.dtype, device=x.device)
        range_t = torch.as_tensor(range_flat, dtype=x.dtype, device=x.device)
        return torch.clamp(2 * (x - center_t) / range_t, -1.0, 1.0)
    else:
        return np.clip(2 * (x - center) / range_flat, -1.0, 1.0)


def read_sde_config(config_path):
    """
    Thin wrapper around read_config that additionally merges in the
    top-level "diffusion" block (nu, diffusion_timesteps, subsample_ratio,
    chunk_size, cache_latents, ...) used by the training-free score
    estimator, so the result can be passed straight to
    due.models.sde_diffusion.generate_labeled_data.
    """
    conf_data, conf_net, conf_train = read_config(config_path)

    raw = safe_load(Path(config_path).read_text())
    conf_train.update(raw.get("diffusion", {}))

    return conf_data, conf_net, conf_train


def sde_evaluate(prediction, truth, save_path, dt=0.01, n_paths=30):
    """
    Evaluates a batch of generated SDE trajectories against the ground truth.

    Unlike due.utils.ode_evaluate (pointwise error), SDE trajectories diverge
    pathwise even for a perfect model, so the comparison is distributional.
    Produces a 2x2 figure: top row is true vs. generated "spaghetti" plots of
    sample paths; bottom-left is ensemble mean with +-1 StdDev and 5th-95th
    percentile bands; bottom-right is absolute error in mean/std over time.

    prediction, truth: (N, dim, T) unnormalized trajectories.
    n_paths: number of individual sample paths to draw in the spaghetti plots.

    Returns a dict of summary metrics (mean_abs_error_mean/std, final_abs_error_mean/std).
    """
    assert prediction.shape[1] == truth.shape[1]
    os.makedirs(save_path, exist_ok=True)

    t_steps = truth.shape[2]
    time_axis = np.arange(t_steps) * dt
    n_paths = min(n_paths, prediction.shape[0], truth.shape[0])

    true_mean = np.mean(truth[:, 0, :], axis=0)
    true_std = np.std(truth[:, 0, :], axis=0)
    pred_mean = np.mean(prediction[:, 0, :], axis=0)
    pred_std = np.std(prediction[:, 0, :], axis=0)
    true_p05, true_p95 = np.percentile(truth[:, 0, :], [5, 95], axis=0)
    pred_p05, pred_p95 = np.percentile(prediction[:, 0, :], [5, 95], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].set_title(f"True Trajectories ({n_paths} samples)")
    for i in range(n_paths):
        axes[0, 0].plot(time_axis, truth[i, 0, :], color='blue', alpha=0.25, linewidth=0.8)
    axes[0, 0].plot(time_axis, true_mean, color='black', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Physical State X")
    axes[0, 0].legend()

    axes[0, 1].set_title(f"Generated Trajectories ({n_paths} samples)")
    for i in range(n_paths):
        axes[0, 1].plot(time_axis, prediction[i, 0, :], color='red', alpha=0.25, linewidth=0.8)
    axes[0, 1].plot(time_axis, pred_mean, color='black', linewidth=2, label='Mean')
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Physical State X")
    axes[0, 1].legend()

    ylim = (min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0]),
            max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1]))
    axes[0, 0].set_ylim(ylim)
    axes[0, 1].set_ylim(ylim)

    axes[1, 0].set_title("Ensemble Spread (±1 StdDev, 5th-95th pct.)")
    axes[1, 0].fill_between(time_axis, true_p05, true_p95, color='blue', alpha=0.10, label='True 5th-95th pct.')
    axes[1, 0].fill_between(time_axis, true_mean - true_std, true_mean + true_std, color='blue', alpha=0.25)
    axes[1, 0].plot(time_axis, true_mean, color='blue', label='True Mean')

    axes[1, 0].fill_between(time_axis, pred_p05, pred_p95, color='red', alpha=0.10, label='Generated 5th-95th pct.')
    axes[1, 0].fill_between(time_axis, pred_mean - pred_std, pred_mean + pred_std, color='red', alpha=0.25)
    axes[1, 0].plot(time_axis, pred_mean, color='red', linestyle='--', label='Generated Mean')
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Absolute Error in Mean & StdDev Over Time")
    axes[1, 1].plot(time_axis, np.abs(true_mean - pred_mean), color='purple', label='|Mean error|')
    axes[1, 1].plot(time_axis, np.abs(true_std - pred_std), color='darkorange', label='|StdDev error|')
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/sde_evaluation.png", dpi=150)
    plt.close()
    print(f"Saved Generative Evaluation Plot to: {save_path}/sde_evaluation.png")

    mean_err = np.mean(np.abs(true_mean - pred_mean))
    std_err = np.mean(np.abs(true_std - pred_std))
    np.savetxt(f"{save_path}/distribution_error.csv", np.stack([time_axis, np.abs(true_mean - pred_mean), np.abs(true_std - pred_std)], axis=1),
               header="time,abs_mean_error,abs_std_error", delimiter=",", comments="")
    print(f"Mean absolute error in ensemble mean: {mean_err:.6f} | in ensemble std: {std_err:.6f}")

    return {
        "mean_abs_error_mean": float(mean_err),
        "mean_abs_error_std": float(std_err),
        "final_abs_error_mean": float(np.abs(true_mean - pred_mean)[-1]),
        "final_abs_error_std": float(np.abs(true_std - pred_std)[-1]),
    }


def sde_evaluate_multidim(prediction, truth, save_path, dt=0.01, pair_phase_space=True,
                          max_pairs=6, psd_dim=0):
    """
    Multi-dimensional analogue of sde_evaluate, for SDEs with problem_dim > 1.
    Writes three figures under save_path:
      1. phase_portraits.png -- 2D histogram of each interleaved (x_i, y_i)
         coordinate pair, true vs generated. Skipped if dim is odd.
      2. marginals.png -- per-coordinate stationary histograms, true vs generated.
      3. dynamics.png -- power spectral density of coordinate psd_dim, plus
         per-coordinate ensemble mean/std vs time, true vs generated.

    prediction, truth: (N, dim, T) unnormalized trajectories.

    Returns the same metric keys as sde_evaluate, averaged over coordinates.
    """
    assert prediction.shape[1] == truth.shape[1]
    os.makedirs(save_path, exist_ok=True)

    dim = truth.shape[1]
    T = min(prediction.shape[2], truth.shape[2])
    prediction = prediction[:, :, :T]
    truth = truth[:, :, :T]
    time_axis = np.arange(T) * dt

    if pair_phase_space and dim % 2 == 0:
        n_pairs = min(dim // 2, max_pairs)
        fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 8), squeeze=False)
        for j in range(n_pairs):
            xi, yi = 2 * j, 2 * j + 1
            tx, ty = truth[:, xi, :].ravel(), truth[:, yi, :].ravel()
            px, py = prediction[:, xi, :].ravel(), prediction[:, yi, :].ravel()
            # Shared extent + bins so the two rows are directly comparable.
            xr = [min(tx.min(), px.min()), max(tx.max(), px.max())]
            yr = [min(ty.min(), py.min()), max(ty.max(), py.max())]
            axes[0, j].hist2d(tx, ty, bins=120, range=[xr, yr], cmap="viridis")
            axes[0, j].set_title(f"Osc {j}: TRUE  (x{xi}, x{yi})")
            axes[1, j].hist2d(px, py, bins=120, range=[xr, yr], cmap="viridis")
            axes[1, j].set_title(f"Osc {j}: GENERATED")
            for r in (0, 1):
                axes[r, j].set_xlabel(f"position x{xi}")
                axes[r, j].set_ylabel(f"velocity x{yi}")
        fig.suptitle("Phase-space density (invariant measure): true vs generated", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_path}/phase_portraits.png", dpi=150)
        plt.close()
    else:
        print(f"[sde_evaluate_multidim] dim={dim} is odd (or pairing disabled); "
              f"skipping phase_portraits.png.")

    ncols = min(3, dim)
    nrows = int(np.ceil(dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for c in range(dim):
        ax = axes[c // ncols][c % ncols]
        tvals, pvals = truth[:, c, :].ravel(), prediction[:, c, :].ravel()
        rng = [min(tvals.min(), pvals.min()), max(tvals.max(), pvals.max())]
        ax.hist(tvals, bins=80, range=rng, density=True, color="blue", alpha=0.45, label="True")
        ax.hist(pvals, bins=80, range=rng, density=True, color="red", alpha=0.45, label="Generated")
        ax.set_title(f"coord x{c} marginal")
        ax.legend(fontsize=7)
    for c in range(dim, nrows * ncols):
        axes[c // ncols][c % ncols].axis("off")
    fig.suptitle("Per-coordinate stationary marginals: true vs generated", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path}/marginals.png", dpi=150)
    plt.close()

    true_mean, pred_mean = truth.mean(axis=0), prediction.mean(axis=0)   # (dim, T)
    true_std, pred_std = truth.std(axis=0), prediction.std(axis=0)       # (dim, T)

    freqs = np.fft.rfftfreq(T, d=dt)
    def _avg_psd(arr):  # arr: (N, T)
        a = arr - arr.mean(axis=1, keepdims=True)
        return (np.abs(np.fft.rfft(a, axis=1)) ** 2).mean(axis=0)
    true_psd = _avg_psd(truth[:, psd_dim, :])
    pred_psd = _avg_psd(prediction[:, psd_dim, :])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].semilogy(freqs, true_psd, color="blue", label="True")
    axes[0].semilogy(freqs, pred_psd, color="red", linestyle="--", label="Generated")
    axes[0].set_title(f"Power spectral density (coord x{psd_dim})")
    axes[0].set_xlabel("frequency"); axes[0].set_ylabel("power"); axes[0].legend()

    for c in range(dim):
        axes[1].plot(time_axis, true_mean[c], color="blue", alpha=0.6)
        axes[1].plot(time_axis, pred_mean[c], color="red", linestyle="--", alpha=0.6)
        axes[2].plot(time_axis, true_std[c], color="blue", alpha=0.6)
        axes[2].plot(time_axis, pred_std[c], color="red", linestyle="--", alpha=0.6)
    axes[1].set_title("Ensemble mean vs time (all coords)\nblue=true, red=generated")
    axes[1].set_xlabel("time")
    axes[2].set_title("Ensemble std vs time (all coords)\nblue=true, red=generated")
    axes[2].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(f"{save_path}/dynamics.png", dpi=150)
    plt.close()

    print(f"Saved multi-dim evaluation plots (phase_portraits/marginals/dynamics) to {save_path}")

    mean_err = float(np.mean(np.abs(true_mean - pred_mean)))
    std_err = float(np.mean(np.abs(true_std - pred_std)))
    np.savetxt(f"{save_path}/distribution_error.csv",
               np.column_stack([time_axis,
                                np.abs(true_mean - pred_mean).mean(axis=0),
                                np.abs(true_std - pred_std).mean(axis=0)]),
               header="time,mean_abs_mean_error_over_dims,mean_abs_std_error_over_dims",
               delimiter=",", comments="")
    print(f"Mean abs error in ensemble mean: {mean_err:.6f} | in ensemble std: {std_err:.6f} (avg over {dim} coords)")

    return {
        "mean_abs_error_mean": mean_err,
        "mean_abs_error_std": std_err,
        "final_abs_error_mean": float(np.mean(np.abs(true_mean - pred_mean)[:, -1])),
        "final_abs_error_std": float(np.mean(np.abs(true_std - pred_std)[:, -1])),
    }