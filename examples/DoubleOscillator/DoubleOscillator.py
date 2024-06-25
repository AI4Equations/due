import due
####################################################
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.ode.ode_dataset(conf_data)
trainX, trainY, test_set, vmin, vmax = data_loader.load("DoubleOscillator_train.mat", "DoubleOscillator_test.mat") # = trainX, trainY, test, vmin, vmax
        
mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)
model = due.models.ODE(trainX, trainY, mynet, conf_train)
model.train()
model.save_hist()

#############################
# doing ensemble averaged prediction
mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)
ensemble_nets = due.networks.fcn.ensemble_resnet(mynet, num_ensembles, save_path)
pred = ensemble_nets.predict(test_set[...,10-conf_data["memory"]:11], test_set.shape[-1]-11, device='cpu')
due.utils.ode_evaluate(prediction=pred, truth=test_set[...,10-conf_data["memory"]:], save_path=conf_train["save_path"])
############
