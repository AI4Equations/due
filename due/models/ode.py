import os
from numpy import savetxt
from scipy.io import savemat
import matplotlib.pyplot as plt
from time import time
from ..utils import *

class ODE:
    """
    Class representing an Ordinary Differential Equation (ODE) model.

    Args:
        trainX (numpy array): The input training data.
        trainY (numpy array): The target training data.
        network: The neural network model.
        config (dict): Configuration parameters for training.

    Attributes:
        trainX (torch.Tensor): The input training data as a PyTorch tensor.
        trainY (torch.Tensor): The target training data as a PyTorch tensor.
        multi_steps (int): The number of steps for multi-step rollout during training.
        memory_steps (int): The number of memory steps for input data.
        device (str): The device (e.g., "cpu", "cuda") to use for training.
        mynet: The neural network model.
        nepochs (int): The number of training epochs.
        bsize (int): The batch size for training.
        lr (float): The initial learning rate for optimization.
        optimizer: The optimizer for model training.
        scheduler: The learning rate scheduler.
        verbose (int): The frequency of printing training progress.
        loss_func: The loss function for training.
        save_path (str): The path to save the trained model.
        do_validation (bool): Flag indicating whether to conduct validation for model selection.
        split (int): The index to split training and validation data.
        validX (torch.Tensor): The input validation data.
        validY (torch.Tensor): The target validation data.
        train_loader: The data loader for training data.
        valid_loader: The data loader for validation data.
        hist (torch.Tensor): The training history.

    Methods:
        train(): Trains the ODE model.
        save_hist(xlog=False, ylog=True): Saves the training history.
        summary(): Prints a summary of the ODE model.
        set_seed(seed): Sets the random seed for reproducibility.
    """
    def __init__(self, trainX, trainY, network, config):
        super().__init__()
        
        self.trainX = torch.from_numpy(trainX)
        self.trainY = torch.from_numpy(trainY)
        
        self.multi_steps = self.trainY.shape[-1]
        if self.trainX.shape[1] > self.trainY.shape[1]:
            self.memory_steps = self.trainX.shape[1]//self.trainY.shape[1]
        self.set_seed(config["seed"])
        self.device = config["device"]

        self.mynet = network.to(self.device)
        self.nepochs = config["epochs"]
        self.bsize   = config["batch_size"]
        
        self.lr      = config["learning_rate"]
        self.optimizer = get_optimizer(config["optimizer"], self.mynet, self.lr)
        self.scheduler = get_schedule(self.optimizer, config["scheduler"], self.nepochs, self.bsize, self.trainX.shape[0])
        self.verbose   = config["verbose"]
        
        self.loss_func = get_loss(config["loss"])
        self.save_path   = config["save_path"]
        try:
            os.mkdir(self.save_path)
        except:
            pass
        
        if config["valid"] > 0:
            self.do_validation = True
            self.hist   = torch.zeros(self.nepochs,2)
            self.split  = int(config["valid"] * self.trainX.shape[0])
            print(self.trainX.shape[0]-self.split, "for training;", self.split, "for validation.")
            self.validX = self.trainX[-self.split:,...]
            self.trainX = self.trainX[:-self.split,...]
            self.validY = self.trainY[-self.split:,...]
            self.trainY = self.trainY[:-self.split,...]
            self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.trainX, self.trainY), batch_size=self.bsize, shuffle=True, drop_last=True)
            self.valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.validX, self.validY), batch_size=self.bsize, shuffle=False, drop_last=True)
        elif config["valid"] <= 0.:
            self.do_validation = False
            self.hist   = torch.zeros(self.nepochs,1)
            self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.trainX, self.trainY), batch_size=self.bsize, shuffle=True, drop_last=True)
        
    def train(self):
        self.summary()
        
        overal_start = time()
        start        = overal_start

        min_loss = 10000000000.0
        for ep in range(self.nepochs):
            self.mynet.train()
            
            train_step = 0.
            for xx, yy in self.train_loader:
                
                xx = xx.to(self.device)
                yy = yy.to(self.device)

                pred = torch.zeros_like(yy)
                for t in range(self.multi_steps):
                    pred[...,t] = self.mynet(xx) #(batch_size, output_dim)
                    xx   = torch.cat((xx[...,self.mynet.output_dim:], pred[...,t]), -1)
                    
                loss       = self.loss_func(yy, pred)
                train_step += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler != None:
                    self.scheduler.step()
                
            train_step /= len(self.train_loader)
            if self.do_validation:
                valid_step = 0.
                for xx, yy in self.valid_loader:
                    
                    xx = xx.to(self.device)
                    yy = yy.to(self.device)

                    pred = torch.zeros_like(yy)
                    for t in range(self.multi_steps):
                        pred[...,t] = self.mynet(xx) #(batch_size, output_dim)
                        xx   = torch.cat((xx[...,self.mynet.output_dim:], pred[...,t]), -1)
                        
                    loss       = self.loss_func(yy, pred)
                    valid_step += loss.item()
                valid_step /= len(self.valid_loader)
                if valid_step < min_loss:
                    torch.save(self.mynet, self.save_path+"/model")
                    min_loss = valid_step
                self.hist[ep,0] = train_step
                self.hist[ep,1] = valid_step
                if (ep+1)%self.verbose ==0:
                    end = time()
                    print(f"Epoch {ep+1} --- Time: {end-start:.2f} seconds --- Training loss: {train_step} --- Validation loss: {valid_step}")
                    start = end
            else:
                if train_step < min_loss:
                    torch.save(self.mynet, self.save_path+"/model")
                    min_loss = train_step
                self.hist[ep,0] = train_step
                
                if (ep+1)%self.verbose ==0:
                    end = time()
                    print(f"Epoch {ep+1} --- Time: {end-start:.2f} seconds --- Training loss: {train_step}")
                    start = end
        
    def save_hist(self, xlog=False, ylog=True):
        savetxt(self.save_path+"/training_history.csv", self.hist)
        
        plt.figure(figsize=(9,9))
        plt.plot(range(1,self.nepochs+1), self.hist[:,0], label="Train")
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.savefig(self.save_path+"/training_history.png")
        plt.close()

    def summary(self):
        """Prints all trainable variables."""
        # TODO: backend tensorflow, pytorch
        print("Number of trainable parameters:", self.mynet.count_params())
        print()
        print("Number of epochs:", self.nepochs)
        print("Batch size:", self.bsize)
        print("The model is trained on "+ self.device)
        
    def set_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True




