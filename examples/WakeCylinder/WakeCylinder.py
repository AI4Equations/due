import due
from numpy import genfromtxt
# Load the configuration for the modules: datasets, networks, and models
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
# Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles
data_loader = due.datasets.pde.pde_dataset(conf_data)
trainX, trainY, coordinates, test_set= data_loader.load("WakeCylinder_train.mat", "WakeCylinder_test.mat")
coordinates_small = genfromtxt("vertices_small.csv", delimiter=",")
# Construct a neural network
mynet = due.networks.transformer.pit(coordinates, coordinates_small, conf_train["device"], conf_net)
#mynet = mynet.load_params(conf_train["save_path"]+"/model")
model = due.models.pde.PDE(trainX, trainY, mynet, conf_train)
model.train()
model.save_hist()
# Conduct long-term prediction for arbitrarily given initial conditions
pred = mynet.predict(test_set[...,:conf_data["memory"]+1], 10, device=conf_train["device"])
elements        = genfromtxt("elements.csv", delimiter=",").astype("int32")-1
due.utils.pde2dirregular_evaluate(coordinates, elements, prediction=pred, truth=test_set, save_path=conf_train["save_path"])