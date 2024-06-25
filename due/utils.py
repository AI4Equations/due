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
        
        
        
        
        
