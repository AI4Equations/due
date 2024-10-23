import due
from network import *
# Load the configuration for the modules: datasets, networks, and models
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
# Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles
data_loader = due.datasets.pde.pde_dataset_osg(conf_data)
trainX, trainY, coordinates, test_set, test_dt, vmin, vmax, tmin, tmax, cmin, cmax= data_loader.load("VorticityOSG_train.mat", "VorticityOSG_test.mat")
# Construct a neural network
mynet = due.networks.fno.osg_fno2d(vmin, vmax, tmin, tmax, conf_net, multiscale=conf_data["multiscale"])# due.networks.fno.osg_fno2d
# Define and train a model, save necessary information of the training history
#mynet = mynet.load_params("model/model")
model = due.models.PDE_osg(trainX, trainY, None, mynet, conf_train)
model.train()
model.save_hist()
# Conduct long-term prediction for arbitrarily given initial conditions
pred = mynet.predict(test_set[...,0], test_dt, device=conf_train["device"])
due.utils.pde2dregular_evaluate(coordinates, prediction=pred, truth=test_set, save_path=conf_train["save_path"])