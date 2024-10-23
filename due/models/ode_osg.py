import numpy as np
from time import time
from ..utils import *
from .ode import ODE

class ODE_osg(ODE):

    def __init__(self, trainX, trainY, osg_data, network, config):
        super(ODE_osg, self).__init__(trainX, trainY, network, config)
        
        self.sg_pairing  = config["sg_pairing"] # non-negative integer
        self.sg_weight   = config["sg_weight"]
        
        self.osg_regularization(osg_data)
            
    def osg_regularization(self, osg_data):
        
        if osg_data==None:
            self.vmin        = self.mynet.vmin
            self.vmax        = self.mynet.vmax
            self.tmin        = self.mynet.tmin
            self.tmax        = self.mynet.tmax
            if self.sg_pairing != 0:
                self.u0_rand     = 2.0*torch.rand(self.trainX.shape[0],self.sg_pairing,self.trainY.shape[1], dtype=self.trainX.dtype)-1.0
                self.dt_rand     = 2.0*torch.rand(self.trainX.shape[0],self.sg_pairing, 2, dtype=self.trainX.dtype)-1.0
                t1_rand          = self.dt_rand[...,:1] * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
                t2_rand          = self.dt_rand[...,1:] * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
                if self.mynet.multiscale:
                    t1_rand = 10 ** (t1_rand)
                    t2_rand = 10 ** (t2_rand)
                t12_rand         = t1_rand + t2_rand
                if self.mynet.multiscale:
                    t12_rand     = torch.log10(t12_rand)
                t12_rand         = 2 * (t12_rand - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)
                self.dt_rand     = torch.concat((self.dt_rand, t12_rand), dim=-1)
                self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.trainX, self.trainY, self.u0_rand, self.dt_rand), batch_size=self.bsize, shuffle=True)
        else:
            self.u0_rand, self.dt_rand = osg_data
            self.u0_rand = torch.from_numpy(self.u0_rand)
            self.dt_rand = torch.from_numpy(self.dt_rand)
            self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.trainX, self.trainY, self.u0_rand, self.dt_rand), batch_size=self.bsize, shuffle=True)
        
    def train(self):
        self.summary()
        self.hist   = torch.zeros(self.nepochs,3)
        overal_start = time()
        start        = overal_start
        
        min_loss = 10000000000.0
        for ep in range(self.nepochs):
            self.mynet.train()
            
            train_step = 0
            loss1_step = 0
            loss2_step = 0
            if self.sg_pairing == 0:
                for xx, yy in self.train_loader:
                    
                    xx = xx.to(self.device)
                    yy = yy.to(self.device)
                    
                    pred = self.mynet(xx) #(batch_size, output_dim)
                        
                    loss1      = self.loss_func(yy, pred)
                    loss       = loss1
                    loss1_step += loss1.item()
                    train_step += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler != None:
                        self.scheduler.step()
                        
            else:
                for xx, yy, uu, tt in self.train_loader:
                    
                    xx = xx.to(self.device)
                    yy = yy.to(self.device)
                    uu = uu.view(-1,self.trainY.shape[1]).to(self.device)
                    tt = tt.view(-1,3).to(self.device)
                    
                    pred    = self.mynet(xx) #(batch_size, output_dim)
                    pred01  = self.mynet(torch.cat((uu, tt[:,0:1]),dim=-1))
                    pred012 = self.mynet(torch.cat((pred01, tt[:,1:2]),dim=-1))
                    pred02  = self.mynet(torch.cat((uu, tt[:,1:2]),dim=-1))
                    pred021 = self.mynet(torch.cat((pred02, tt[:,0:1]),dim=-1))
                    pred2   = self.mynet(torch.cat((uu, tt[:,2:3]),dim=-1))
                        
                        
                    loss1   = self.loss_func(yy, pred)
                    loss2   = 0.5 * (self.loss_func(pred012, pred2) + self.loss_func(pred021, pred2))
                    loss    = (loss1 + self.sg_weight * loss2)/(1.0 + self.sg_weight)
                    loss1_step += loss1.item()
                    loss2_step += loss2.item()
                    train_step += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler != None:
                        self.scheduler.step()
                
            loss1_step /= len(self.train_loader)
            loss2_step /= len(self.train_loader)
            train_step /= len(self.train_loader)
            if train_step < min_loss:
                torch.save(self.mynet, self.save_path+"/model")
                min_loss = train_step
            self.hist[ep,0] = train_step
            self.hist[ep,1] = loss1_step
            self.hist[ep,2] = loss2_step
            
            if (ep+1)%self.verbose ==0:
                end = time()
                print(f"Epoch {ep+1} - Time: {end-start:.2f} seconds - Training loss: {train_step:.10f} - Fitting loss: {loss1_step:.10f} - Sg loss: {loss2_step:.10f}")
                start = end
                
                
                
                
                