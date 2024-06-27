import due
####################################################
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.ode.ode_dataset(conf_data)
trainX, trainY, test_set, vmin, vmax = data_loader.load("DoubleOscillator_train.mat", "DoubleOscillator_test.mat") # = trainX, trainY, test, vmin, vmax
        

mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)
#mynet = mynet.load_params("model/model")
model = due.models.ODE(trainX, trainY, mynet, conf_train)
model.train()
model.save_hist()

#############################
# doing prediction
pred = mynet.predict(test_set[...,10-conf_data["memory"]:11], test_set.shape[-1]-11, device='cpu')
due.utils.ode_evaluate(prediction=pred, truth=test_set[...,10-conf_data["memory"]:], save_path=conf_train["save_path"])
############
