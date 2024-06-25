import due
from network import *

####################################################
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.pde.pde_dataset_osg(conf_data)
trainX, trainY, coordinates, test_set, test_dt, vmin, vmax, tmin, tmax, cmin, cmax= data_loader.load("VorticityOSG_train.mat", "VorticityOSG_test.mat") # = trainX, testX, trainY, testY, vmin, vmax
print(vmin, vmax, tmin, tmax, cmin, cmax)
mynet = due.networks.fno.osg_fno2d(vmin, vmax, tmin, tmax, conf_net, multiscale=conf_data["multiscale"])# due.networks.fno.osg_fno2d


#mynet = mynet.load_params("model/model")
model = due.models.PDE_osg(trainX, trainY, None, mynet, conf_train)
model.train()
model.save_hist()
print(test_set.shape)
pred = mynet.predict(test_set[...,0], test_dt, device=conf_train["device"])
due.utils.pde2dregular_evaluate(coordinates, prediction=pred, truth=test_set, save_path=conf_train["save_path"])
