import numpy as np
np.random.seed(0)
from scipy.io import loadmat

class pde_dataset_osg():
    """
    A class representing a partial differential equation (PDE) dataset.

    Attributes:
        problem_type (str): The type of the PDE problem.
        nbursts (int): The number of short bursts sampled from each trajectory.
        multiscale (bool): Indicates whether the dataset exhibits multiscale property in time.
        dtype (str): The data type of the dataset.

    Methods:
        load(file_path_train, file_path_test): Loads the PDE dataset from the given file paths.
        normalize(data, dt, coords): Normalizes the PDE dataset.

    """
    def __init__(self, config):
        
        self.problem_type  = config["problem_type"]
        self.nbursts       = config["nbursts"]
        self.multiscale    = config["multiscale"]
        self.dtype         = config["dtype"]
        
    def load(self, file_path_train, file_path_test):
        """
        Loads the PDE dataset from the given file paths.

        Args:
            file_path_train (str): The file path of the training dataset.
            file_path_test (str or None): The file path of the test dataset.

        Returns:
            tuple: A tuple containing the loaded dataset and other related information.
        """
    
        try:
            data = loadmat(file_path_train)
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
        print("Input shape {}.".format(target_X.shape), "Output shape {}.".format(target_Y.shape))

        if file_path_test==None:
            return target_X.astype(self.dtype), target_Y.astype(self.dtype), coords.astype(self.dtype), dt.astype(self.dtype), vmin.astype(self.dtype), vmax.astype(self.dtype), tmin.astype(self.dtype), tmax.astype(self.dtype), cmin.astype(self.dtype), cmax.astype(self.dtype)
        else:
            ## Load test set without normalization
            data = loadmat(file_path_test)
            dt   = data["dt"]
            data = data["trajectories"]
            return target_X.astype(self.dtype), target_Y.astype(self.dtype), coords.astype(self.dtype), data.astype(self.dtype), dt.astype(self.dtype), vmin.astype(self.dtype), vmax.astype(self.dtype), tmin.astype(self.dtype), tmax.astype(self.dtype), cmin.astype(self.dtype), cmax.astype(self.dtype)
        
    def normalize(self, data, dt, coords):
        """
        Normalizes the PDE dataset.

        Args:
            data (ndarray): The data array of the dataset.
            dt (ndarray): The time step array of the dataset.
            coords (ndarray): The coordinates array of the dataset.

        Returns:
            tuple: A tuple containing the normalized dataset and other related information.

        """
        axes      = tuple(np.delete(np.arange(len(data.shape)),[-2]))
        vmax      = np.max(data, axis=axes, keepdims=True)
        vmin      = np.min(data, axis=axes, keepdims=True)
        data      = 2*(data-0.5*(vmax+vmin))/(vmax-vmin)
        data      = np.clip(data, -1, 1)
        
        if self.multiscale:
            dt = np.log10(dt)
        tmax      = dt.max()
        tmin      = dt.min()
        dt        = 2*(dt-0.5*(tmax+tmin))/(tmax-tmin)
        dt        = np.clip(dt, -1, 1)
        
        
        cmax      = np.max(coords, axis=tuple(range(len(coords.shape[:-1]))), keepdims=True)
        cmin      = np.min(coords, axis=tuple(range(len(coords.shape[:-1]))), keepdims=True)
        coords    = 2*(coords-0.5*(cmax+cmin))/(cmax-cmin)
        coords    = np.clip(coords, -1, 1)
        return data, dt, coords, vmin, vmax, tmin, tmax, cmin, cmax
   
class pde_dataset():
    """
    A class representing a PDE dataset.

    Parameters:
    - config (dict): A dictionary containing the configuration parameters for the dataset.

    Attributes:
    - problem_type (str): The type of the problem (e.g., "1d_regular", "2d_irregular").
    - memory_steps (int): The number of memory embedding steps.
    - multi_steps (int): The number of steps in multi-step loss.
    - nbursts (int): The number of short bursts sampled from each tracjectory.
    - dtype (str): The data type.
    """

    def __init__(self, config):
        self.problem_type = config["problem_type"]
        self.memory_steps = config["memory"]
        self.multi_steps = config["multi_steps"]
        self.nbursts = config["nbursts"]
        self.dtype = config["dtype"]

        assert self.memory_steps >= 0

    def load(self, file_path_train, file_path_test):
        """
        Load the dataset from the given file paths.

        Parameters:
        - file_path_train (str): The file path for the training dataset.
        - file_path_test (str or None): The file path for the test dataset.

        Returns:
        - trainX (ndarray): The input training data.
        - trainY (ndarray): The output training data.
        - coords (ndarray): The coordinates data.
        - data_test (ndarray): The test data.
        """
        try:
            data = loadmat(file_path_train)
            try:
                coords = data["coordinates"]
                data = data["trajectories"]
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

        # Rest of the code...

        if file_path_test == None:
            return trainX.astype(self.dtype), trainY.astype(self.dtype), coords.astype(self.dtype)
        else:
            # Load test set without normalization
            data_test = loadmat(file_path_test)
            data_test = data_test["trajectories"]
            return trainX.astype(self.dtype), trainY.astype(self.dtype), coords.astype(self.dtype), data_test.astype(self.dtype)