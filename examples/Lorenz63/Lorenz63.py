import due
####################################################
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.ode.ode_dataset(conf_data)
trainX, trainY, test_set, vmin, vmax = data_loader.load("Lorenz63_train.mat", "Lorenz63_test.mat")

## train a prior model
affine = due.networks.fcn.affine(vmin, vmax, conf_net)
prior_model = due.models.ODE(trainX, trainY, affine, conf_train)
prior_model.train()
prior_model.save_hist()

## train the correction model
# mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)
mynet = due.networks.fcn.gresnet(prior_model.mynet, vmin, vmax, conf_net)
conf_train['save_path'] = "./model"
conf_train['batch_size'] = 5
# mynet = mynet.load_params("model/model")
model = due.models.ODE(trainX, trainY, mynet, conf_train)
model.train()
model.save_hist()

pred = mynet.predict(test_set[...,:conf_data["memory"]+1], 10000, device='cpu')
due.utils.ode_evaluate(prediction=pred, truth=test_set, save_path=conf_train["save_path"])
############
