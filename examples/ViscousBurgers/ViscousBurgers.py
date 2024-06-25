import due
####################################################
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.pde.pde_dataset(conf_data)
trainX, trainY, coordinates, test_set= data_loader.load("ViscousBurgers_train.mat", "ViscousBurgers_test.mat") # = trainX, testX, trainY, testY, vmin, vmax
## Project to modal space
reducer = due.utils.generalized_fourier_projection1d(coordinates, conf_train)
trainX_modal, trainY_modal, vmin, vmax = reducer.forward(trainX, trainY, training=True)

## train
mynet = due.networks.fcn.resnet(vmin, vmax, conf_net)
#mynet = mynet.load_params("model/model")

model = due.models.ODE(trainX_modal, trainY_modal, mynet, conf_train)
model.train()
model.save_hist()

testX_modal, testY_modal = reducer.forward(test_set[...,:conf_data["memory"]+1], test_set[...,conf_data["memory"]+1:], training=False)
pred_modal = mynet.predict(testX_modal, 200, device=conf_train["device"])
pred = reducer.backward(pred_modal)
due.utils.pde1d_evaluate(coordinates, prediction=pred, truth=test_set, save_path=conf_train["save_path"])
