import due
# Load the configuration for the modules: datasets, networks, and models
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
# Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles
data_loader = due.datasets.pde.pde_dataset(conf_data)
trainX, trainY, coordinates, test_set= data_loader.load("ViscousBurgers_train.mat", "ViscousBurgers_test.mat")
# Project PDE data to modal space
reducer = due.utils.generalized_fourier_projection1d(coordinates, conf_train)
trainX_modal, trainY_modal, vmin, vmax = reducer.forward(trainX, trainY, training=True)
# Construct a neural network
mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)
# Define and train a model, save necessary information of the training history
#mynet = mynet.load_params("model/model")
model = due.models.ODE(trainX_modal, trainY_modal, mynet, conf_train)
model.train()
model.save_hist()
# Conduct long-term prediction for arbitrarily given initial conditions
testX_modal, testY_modal = reducer.forward(test_set[...,:conf_data["memory"]+1], test_set[...,conf_data["memory"]+1:], training=False)
pred_modal = mynet.predict(testX_modal, 200, device=conf_train["device"])
pred = reducer.backward(pred_modal)
due.utils.pde1d_evaluate(coordinates, prediction=pred, truth=test_set, save_path=conf_train["save_path"])