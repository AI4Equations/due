import due
# Load the configuration for the modules: datasets, networks, and models
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
# Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles
data_loader = due.datasets.ode.ode_dataset(conf_data)
trainX, trainY, test_set, vmin, vmax = data_loader.load("DoubleOscillator_train.mat", "DoubleOscillator_test.mat")
# Construct a neural network
mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)
# Define and train a model, save necessary information of the training history
#mynet = mynet.load_params("model/model")
model = due.models.ODE(trainX, trainY, mynet, conf_train)
model.train()
model.save_hist()
# Conduct long-term prediction for arbitrarily given initial conditions
pred = mynet.predict(test_set[...,10-conf_data["memory"]:11], test_set.shape[-1]-11, device='cpu')
due.utils.ode_evaluate(prediction=pred, truth=test_set[...,10-conf_data["memory"]:], save_path=conf_train["save_path"])
