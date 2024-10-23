import os
from numpy import savetxt
from scipy.io import savemat
import matplotlib.pyplot as plt
from time import time
from ..utils import *


class ODE:

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
        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.trainX, self.trainY), batch_size=self.bsize, shuffle=True)
        
    def train(self):
        self.summary()
        self.hist   = torch.zeros(self.nepochs,1)
        overal_start = time()
        start        = overal_start

        min_loss = 10000000000.0
        for ep in range(self.nepochs):
            self.mynet.train()
            
            train_step = 0
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




