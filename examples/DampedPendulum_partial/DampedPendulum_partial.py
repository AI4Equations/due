import yaml
from pathlib import Path

import due
####################################################
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.ode.ode_dataset(conf_data)
trainX, trainY, test_set, vmin, vmax = data_loader.load("DampedPendulum_partial_train.mat", "DampedPendulum_partial_test.mat") # = trainX, testX, trainY, testY, vmin, vmax
mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)

#mynet = mynet.load_params("model/model")
model = due.models.ODE(trainX, trainY, mynet, conf_train)
model.train()
model.save_hist()

pred = mynet.predict(test_set[...,:conf_data["memory"]+1], test_set.shape[-1]-conf_data["memory"]-1, device='cpu')
due.utils.ode_evaluate(prediction=pred, truth=test_set, save_path=conf_train["save_path"])

############
