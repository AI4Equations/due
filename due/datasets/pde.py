import numpy as np
np.random.seed(0)
from scipy.io import loadmat
    
class pde_dataset_osg():
    def __init__(self, config):
        
        self.problem_type  = config["problem_type"]
        self.nbursts       = config["nbursts"]
        self.multiscale    = config["multiscale"]
        self.dtype         = config["dtype"]
        
    def load(self, file_path_train, file_path_test):
    
        try:
            data = loadmat(file_path_train) #(10000,2,1001) #np.genfromtxt(file_path, delimiter=",")
        except NotImplementedError:
            print("Your mat file is too large. Be patient.")
            import mat73
            data = mat73.loadmat(file_path_train)
        
        dt  = data["dt"]
        try:
            coords = data["coordinates"]
            data   = data["trajectories"]
        except:
            raise ValueError("Please name your dataset as trajectories.")
        assert len(data.shape) >= 4
        
        if self.problem_type == "1d_regular":
            
            N   = data.shape[0]
            L   = data.shape[1]
            assert coords.shape == (L,) or coords.shape == (L,1) or coords.shape == (1,L)
            coords = coords.reshape(L,1)
            D   = data.shape[2]
            T   = data.shape[3]
            print("One-dimensional regular dataset loaded, {} trajectories, {} grid points, {} variables, {} time instances".format(N,L,D,T))
            
        elif self.problem_type == "2d_regular":
            N   = data.shape[0]
            H   = data.shape[1]
            W   = data.shape[2]
            assert coords.shape == (H,W,2)
            D   = data.shape[3]
            T   = data.shape[4]
            print("Two-dimensional regular dataset loaded, {} trajectories, {} rows, {} columns, {} variables, {} time instances".format(N,H,W,D,T))
            
        elif self.problem_type == "1d_irregular":
            N   = data.shape[0]
            L   = data.shape[1]
            assert coords.shape == (L,) or coords.shape == (L,1) or coords.shape == (1,L)
            coords = coords.reshape(L,1)
            D   = data.shape[2]
            T   = data.shape[3]
            print("One-dimensional irregular dataset loaded, {} trajectories, {} collocation points, {} variables, {} time instances".format(N,L,D,T))
            
        elif self.problem_type == "2d_irregular":
            N   = data.shape[0]
            L   = data.shape[1]
            assert coords.shape == (L,2)
            D   = data.shape[2]
            T   = data.shape[3]
            print("Two-dimensional irregular dataset loaded, {} trajectories, {} collocation points, {} variables, {} time instances".format(N,L,D,T))
            
        else:
            raise ValueError("1D and 2D data collected on either uniform grids or unstructured meshes are supported. Make sure that your dataset is correctly organized. 3D problems are not yet supported")
            
        ###### normalization
        data, dt, coords, vmin, vmax, tmin, tmax, cmin, cmax = self.normalize(data, dt, coords)
        if self.nbursts > T-1:
            self.nbursts = T-1
        if self.problem_type in ["1d_regular", "1d_irregular", "2d_irregular"]:
            target_X  = np.zeros((N*self.nbursts, L, D))
            target_dt = np.zeros((N*self.nbursts, 1))
            target_Y  = np.zeros((N*self.nbursts, L, D))
        else:
            target_X  = np.zeros((N*self.nbursts, H, W, D))
            target_dt = np.zeros((N*self.nbursts, 1))
            target_Y  = np.zeros((N*self.nbursts, H, W, D))
        
        for i in range(N):
            if self.nbursts < T-1:
                inits    = np.random.randint(0, T-1, self.nbursts)#(0, K-subset+1, J0)
                while np.unique(inits).shape != inits.shape:
                    inits    = np.random.randint(0, T-1, self.nbursts)
            else:
                inits = np.arange(T-1)
            selected_X  = np.asarray([data[i,...,init] for init in inits])
            selected_dt = np.asarray([dt[i,init] for init in inits])
            selected_Y  = np.asarray([data[i,...,init+1] for init in inits])
            target_X[i*self.nbursts:(i+1)*self.nbursts,...]  = selected_X
            target_dt[i*self.nbursts:(i+1)*self.nbursts,...] = selected_dt[:,np.newaxis]
            target_Y[i*self.nbursts:(i+1)*self.nbursts,...]  = selected_Y 
        print("Dataset regrouped, {} bursts.".format(target_Y.shape[0]))
            
        #######################################################
        if self.problem_type in ["1d_regular", "1d_irregular", "2d_irregular"]:
            target_dt = np.tile(target_dt[:,np.newaxis,:], [1,L,1]) #(N,L,1)
            target_X  = np.concatenate((target_X,target_dt), axis=-1) # (N,L,D+1)
        else:
            target_dt = np.tile(target_dt[:,np.newaxis,np.newaxis,:], [1,H,W,1]) #(N,H,W,1)
            target_X  = np.concatenate((target_X,target_dt), axis=-1) # (N,H,W,D+1)
        ## Load test set without normalization
        data = loadmat(file_path_test)
        dt   = data["dt"]
        data = data["trajectories"]
        print("Input shape {}.".format(target_X.shape), "Output shape {}.".format(target_Y.shape))
        return target_X.astype(self.dtype), target_Y.astype(self.dtype), coords.astype(self.dtype), data.astype(self.dtype), dt.astype(self.dtype), vmin.astype(self.dtype), vmax.astype(self.dtype), tmin.astype(self.dtype), tmax.astype(self.dtype), cmin.astype(self.dtype), cmax.astype(self.dtype)
        
    def normalize(self, data, dt, coords):
    
        axes      = tuple(np.delete(np.arange(len(data.shape)),[-2]))
        vmax      = np.max(data, axis=axes, keepdims=True)
        vmin      = np.min(data, axis=axes, keepdims=True)
        data      = 2*(data-0.5*(vmax+vmin))/(vmax-vmin)
        
        if self.multiscale:
            dt = np.log10(dt)
        tmax      = dt.max()
        tmin      = dt.min()
        dt        = 2*(dt-0.5*(tmax+tmin))/(tmax-tmin)
        
        
        cmax      = np.max(coords, axis=tuple(range(len(coords.shape[:-1]))), keepdims=True)
        cmin      = np.min(coords, axis=tuple(range(len(coords.shape[:-1]))), keepdims=True)
        coords    = 2*(coords-0.5*(cmax+cmin))/(cmax-cmin)
        return data, dt, coords, vmin, vmax, tmin, tmax, cmin, cmax
        ############################
        
   
class pde_dataset():
    def __init__(self, config):
        
        self.problem_type  = config["problem_type"]
        self.memory_steps  = config["memory"]
        self.multi_steps   = config["multi_steps"]
        self.nbursts       = config["nbursts"]
        self.dtype         = config["dtype"]
        
        assert self.memory_steps >= 0
        
    def load(self, file_path_train, file_path_test):
    
        try:
            data = loadmat(file_path_train) #(10000,2,1001) #np.genfromtxt(file_path, delimiter=",")
            try:
                coords = data["coordinates"]
                data   = data["trajectories"]#[:,:,:,np.newaxis,:]
            except:
                raise ValueError("Please name your dataset as trajectories.")
        except NotImplementedError:
            print("Your mat file is too large. Be patient.")
            import h5py
            with h5py.File(file_path_train, 'r') as f:
                try:
                    coords = f["coordinates"][:].T
                    data = f["trajectories"][:].T
                except:
                    raise ValueError("Please name your dataset as trajectories.")
        
        print(data.shape, coords.shape)
        assert len(data.shape) >= 4
        
        if self.problem_type == "1d_regular":
            
            N   = data.shape[0]
            L   = data.shape[1]
            assert coords.shape == (L,) or coords.shape == (L,1) or coords.shape == (1,L)
            coords = coords.reshape(L,1)
            D   = data.shape[2]
            T   = data.shape[3]
            print("One-dimensional regular dataset loaded, {} trajectories, {} grid points, {} variables, {} time instances".format(N,L,D,T))
            
        elif self.problem_type == "2d_regular":
            N   = data.shape[0]
            H   = data.shape[1]
            W   = data.shape[2]
            assert coords.shape == (H,W,2)
            D   = data.shape[3]
            T   = data.shape[4]
            print("Two-dimensional regular dataset loaded, {} trajectories, {} rows, {} columns, {} variables, {} time instances".format(N,H,W,D,T))
            
        elif self.problem_type == "1d_irregular":
            N   = data.shape[0]
            L   = data.shape[1]
            assert coords.shape == (L,) or coords.shape == (L,1) or coords.shape == (1,L)
            coords = coords.reshape(L,1)
            D   = data.shape[2]
            T   = data.shape[3]
            print("One-dimensional irregular dataset loaded, {} trajectories, {} collocation points, {} variables, {} time instances".format(N,L,D,T))
            
        elif self.problem_type == "2d_irregular":
            N   = data.shape[0]
            L   = data.shape[1]
            assert coords.shape == (L,2)
            D   = data.shape[2]
            T   = data.shape[3]
            print("Two-dimensional irregular dataset loaded, {} trajectories, {} collocation points, {} variables, {} time instances".format(N,L,D,T))
            
        else:
            raise ValueError("1D and 2D data collected on either uniform grids or unstructured meshes are supported. Make sure that your dataset is correctly organized. 3D problems are not yet supported")
            
        if self.nbursts > T-self.memory_steps-self.multi_steps-1:
            self.nbursts = T-self.memory_steps-self.multi_steps-1
        if self.problem_type in ["1d_regular", "1d_irregular", "2d_irregular"]:
            target = np.zeros((N*self.nbursts, L, D, self.memory_steps+self.multi_steps+2))
        else:
            target = np.zeros((N*self.nbursts, H, W, D, self.memory_steps+self.multi_steps+2))
            
        for i in range(N):
            if self.nbursts < T-self.memory_steps-self.multi_steps-1:
                inits    = np.random.randint(0, T-self.memory_steps-self.multi_steps-1, self.nbursts)#(0, K-subset+1, J0)
                while np.unique(inits).shape != inits.shape:
                    inits    = np.random.randint(0, T-self.memory_steps-self.multi_steps-1, self.nbursts)
            else:
                inits = np.arange(T-self.memory_steps-self.multi_steps-1)
            selected = np.asarray([data[i,...,init:init+self.memory_steps+self.multi_steps+2] for init in inits])
            target[i*self.nbursts:(i+1)*self.nbursts,...] = selected
        print("Dataset regrouped, {} bursts, {} time instances".format(target.shape[0], target.shape[-1]))
            
        #######################################################
        ###### normalization
        np.random.shuffle(target)
        np.random.shuffle(target)
        np.random.shuffle(target)
        np.random.shuffle(target)
        np.random.shuffle(target)
        
        trainX = target[...,:self.memory_steps+1]
        trainY = target[...,self.memory_steps+1:]
        ## Load test set without normalization
        data_test = loadmat(file_path_test)
        data_test = data_test["trajectories"]
        print(target.shape, data_test.shape)
        return trainX.astype(self.dtype), trainY.astype(self.dtype), coords.astype(self.dtype), data_test.astype(self.dtype)
        
        ############################


