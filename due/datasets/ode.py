import numpy as np
np.random.seed(0)
from scipy.io import loadmat

class ode_dataset_osg():
    """
    A class representing an ODE dataset with varied time lags.

    Args:
        config (dict): A dictionary containing the configuration parameters for the dataset.
            - problem_dim (int): The number of state variables.
            - nbursts (int): The number of short bursts sampled from each trajectory.
            - multiscale (bool): Flag indicating whether to use multiscale normalization.

    Methods:
        load(file_path_train, file_path_test=None):
            Loads the dataset from the given file paths and returns the training data.

        normalize(data_X, data_dt, data_Y):
            Normalizes the input and output data.
    """
    def __init__(self, config):
        
        self.problem_dim   = config["problem_dim"]
        self.nbursts       = config["nbursts"]
        self.multiscale    = config["multiscale"]
        
    def load(self, file_path_train, file_path_test):
        """
        Loads the dataset from the given file paths and returns the training data.

        Args:
            file_path_train (str): The file path to the training data.
            file_path_test (str or None): The file path to the test data.

        Returns:
            tuple: A tuple containing the training data and other information.
                - trainX (ndarray): The input training data.
                - trainY (ndarray): The output training data.
                - dt (ndarray): The time step data.
                - vmin (ndarray): The minimum values used for normalization.
                - vmax (ndarray): The maximum values used for normalization.
                - tmin (float): The minimum time step size value used for normalization.
                - tmax (float): The maximum time step size value used for normalization.
        """
        try:
            data = loadmat(file_path_train)
        except NotImplementedError:
            print("Your mat file is too large. Be patient.")
            import mat73
            data = mat73.loadmat(file_path_train)
        dt  = data["dt"]
        try:
            data = data["trajectories"]
        except:
            raise ValueError(f"Please name your dataset as trajectories.")
        
        
        N   = data.shape[0]
        T   = data.shape[2]
        if data.shape[1] != self.problem_dim:
            raise ValueError('Only support data arrays with size (N,d,T), N being number of trajectories, d being the number of state variables, T being the number of time instances.')
        print("Dataset loaded, {} trajectories, {} variables, {} time instances".format(N,self.problem_dim,T))

        # split dataset
        if self.nbursts >= T-1:
            self.nbursts = T-1
        target_X  = np.zeros((N*self.nbursts, self.problem_dim))
        target_dt = np.zeros((N*self.nbursts, 1))
        target_Y  = np.zeros((N*self.nbursts, self.problem_dim))
        for i in range(N):
            if self.nbursts < T-1:
                inits    = np.random.randint(0, T-1, self.nbursts)#(0, K-subset+1, J0)
                while  np.unique(inits).shape != inits.shape:
                    inits    = np.random.randint(0, T-1, self.nbursts)
            elif self.nbursts == T-1:
                inits = np.arange(T-1)
            selected_X  = np.asarray([data[i,:,init] for init in inits])
            selected_dt = np.asarray([dt[i,init] for init in inits])
            selected_Y  = np.asarray([data[i,:,init+1] for init in inits])
            target_X[i*self.nbursts:(i+1)*self.nbursts,...]  = selected_X
            target_dt[i*self.nbursts:(i+1)*self.nbursts,...] = selected_dt[:,np.newaxis]
            target_Y[i*self.nbursts:(i+1)*self.nbursts,...]  = selected_Y
        print("Dataset regrouped, {} bursts, {} variables".format(target_Y.shape[0],self.problem_dim))

        ###### normalize data into [-1,1]
        target_X, target_dt, target_Y, vmin, vmax, tmin, tmax = self.normalize(target_X, target_dt, target_Y)
        print("Training data is normalized")
        ###########################################
        ########## input output pairing
        trainX = np.hstack((target_X, target_dt))
        trainY = target_Y
        print("Input shape {}.".format(trainX.shape), "Output shape {}.".format(trainY.shape))
        ################
        if file_path_test==None:
            return trainX, trainY, dt, vmin, vmax, tmin, tmax
        else:
            ## Load test set without normalization
            data = loadmat(file_path_test)
            dt   = data["dt"]
            data = data["trajectories"]
            return trainX, trainY, data, dt, vmin, vmax, tmin, tmax

    def normalize(self, data_X, data_dt, data_Y):
        """
        Normalizes the input and output data.

        Args:
            data_X (ndarray): The input data.
            data_dt (ndarray): The time step data.
            data_Y (ndarray): The output data.

        Returns:
            tuple: A tuple containing the normalized data and other information.
                - target_X (ndarray): The normalized input data.
                - target_dt (ndarray): The normalized time step data.
                - target_Y (ndarray): The normalized output data.
                - vmin (ndarray): The minimum values used for normalization.
                - vmax (ndarray): The maximum values used for normalization.
                - tmin (float): The minimum time step value used for normalization.
                - tmax (float): The maximum time step value used for normalization.
        """
        vmax      = np.maximum(np.max(data_X, axis=0, keepdims=True), np.max(data_Y, axis=0, keepdims=True))
        vmin      = np.minimum(np.min(data_X, axis=0, keepdims=True), np.min(data_Y, axis=0, keepdims=True))
        target_X  = 2*(data_X-0.5*(vmax+vmin))/(vmax-vmin)
        target_Y  = 2*(data_Y-0.5*(vmax+vmin))/(vmax-vmin)
        target_X  = np.clip(target_X, -1, 1)
        target_Y  = np.clip(target_Y, -1, 1)
        
        if self.multiscale:
            data_dt = np.log10(data_dt)
        tmax      = data_dt.max()
        tmin      = data_dt.min()
        target_dt = 2*(data_dt-0.5*(tmax+tmin))/(tmax-tmin)
        target_dt = np.clip(target_dt, -1, 1)
        return target_X, target_dt, target_Y, vmin, vmax, tmin, tmax         
        
class ode_dataset():
    """
    A class representing an ODE dataset.

    Args:
        config (dict): A dictionary containing configuration parameters for the dataset.

    Attributes:
        problem_dim (int): The number of state variables.
        memory_steps (int): The number of memory embeddings.
        multi_steps (int): The number of steps for the multi-step loss.
        nbursts (int): The number of short bursts sampled from each trajectory.
        dtype (str): The data type.

    Methods:
        load(file_path_train, file_path_test): Loads the dataset from the given file paths.
        normalize(data): Normalizes the data into the range [-1, 1].
    """

    def __init__(self, config):
        self.problem_dim = config["problem_dim"]
        self.memory_steps = config["memory"]
        self.multi_steps = config["multi_steps"]
        self.nbursts = config["nbursts"]
        self.dtype = config["dtype"]
        
        assert self.memory_steps >= 0
                
    def load(self, file_path_train, file_path_test):
        """
        Loads the dataset from the given file paths.

        Args:
            file_path_train (str): The file path of the training data.
            file_path_test (str or None): The file path of the test data.

        Returns:
            tuple: A tuple containing the training input, training output, test data (if provided), minimum values, and maximum values.
        """
        try:
            data = loadmat(file_path_train)
        except NotImplementedError:
            print("Your mat file is too large. Be patient.")
            import mat73
            data = mat73.loadmat(file_path_train)
        try:
            data = data["trajectories"]
        except:
            raise ValueError("Please name your dataset as trajectories.")

        N = data.shape[0]
        T = data.shape[2]
        if data.shape[1] != self.problem_dim:
            raise ValueError('Only support data arrays with size (N,d,T), N being number of trajectories, d being the number of state variables, T being the number of time instances.')
        print("Dataset loaded, {} trajectories, {} variables, {} time instances".format(N, self.problem_dim, T))

        if self.nbursts > T - self.multi_steps - self.memory_steps - 1:
            self.nbursts = T - self.multi_steps - self.memory_steps - 1
            target = np.zeros((N * self.nbursts, self.problem_dim, self.memory_steps + self.multi_steps + 2))
            inits = np.arange(T - self.multi_steps - self.memory_steps - 1)
            for i in range(N):
                selected = np.asarray([data[i, :, init:init + self.memory_steps + self.multi_steps + 2] for init in inits])
                target[i * self.nbursts:(i + 1) * self.nbursts, ...] = selected
        else:
            target = np.zeros((N * self.nbursts, self.problem_dim, self.memory_steps + self.multi_steps + 2))
            for i in range(N):
                inits = np.random.randint(0, T - self.multi_steps - self.memory_steps - 1, self.nbursts)
                while np.unique(inits).shape != inits.shape:
                    inits = np.random.randint(0, T - self.multi_steps - self.memory_steps - 1, self.nbursts)
                selected = np.asarray([data[i, :, init:init + self.memory_steps + self.multi_steps + 2] for init in inits])
                target[i * self.nbursts:(i + 1) * self.nbursts, ...] = selected

        print("Dataset regrouped, {} bursts, {} variables, {} time instances".format(target.shape[0], self.problem_dim, target.shape[2]))
        
        ###### normalize data into [-1,1]
        np.random.shuffle(target)
        np.random.shuffle(target)
        np.random.shuffle(target)
        np.random.shuffle(target)
        np.random.shuffle(target)
        target, vmin, vmax = self.normalize(target)
        print("Training data is normalized")
        ###########################################
        ########## input output pairing
        trainX = target[..., :self.memory_steps + 1].transpose(0, 2, 1).reshape(target.shape[0], -1)
        trainY = target[..., self.memory_steps + 1:]
        print("Input shape {}.".format(trainX.shape), "Output shape {}.".format(trainY.shape))

        if file_path_test == None:
            return trainX.astype(self.dtype), trainY.astype(self.dtype), np.asarray(vmin).astype(self.dtype), np.asarray(vmax).astype(self.dtype)
        else:
            ## Load test set without normalization
            data = loadmat(file_path_test)
            data = data["trajectories"]
            return trainX.astype(self.dtype), trainY.astype(self.dtype), data.astype(self.dtype), np.asarray(vmin).astype(self.dtype), np.asarray(vmax).astype(self.dtype)

    def normalize(self, data):
        """
        Normalize the data into the range [-1, 1].

        Args:
            data (ndarray): The data to be normalized.

        Returns:
            tuple: A tuple containing the normalized data, minimum values, and maximum values.
        """
        vmax = np.max(data, axis=(0, 2), keepdims=True)
        vmin = np.min(data, axis=(0, 2), keepdims=True)
        target = 2 * (data - 0.5 * (vmax + vmin)) / (vmax - vmin)
        target = np.clip(target, -1, 1)
        return target, vmin, vmax
            
