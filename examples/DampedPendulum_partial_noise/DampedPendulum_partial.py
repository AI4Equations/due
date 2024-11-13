import due
# Load the configuration for the modules: datasets, networks, and models
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.ode.ode_dataset(conf_data)
# Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles
trainX, trainY, test_set, vmin, vmax = data_loader.load("DampedPendulum_partial_train.mat", "DampedPendulum_partial_test.mat")
# Construct a neural network
mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)
# Define and train a model, save necessary information of the training history
#mynet = mynet.load_params("model/model")
model = due.models.ODE(trainX, trainY, mynet, conf_train)
model.train()
model.save_hist()
# Conduct long-term prediction for arbitrarily given initial conditions
pred = mynet.predict(test_set[...,:conf_data["memory"]+1], test_set.shape[-1]-conf_data["memory"]-1, device='cpu')
due.utils.ode_evaluate(prediction=pred, truth=test_set, save_path=conf_train["save_path"])