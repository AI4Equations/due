import due
from numpy import genfromtxt
####################################################
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.pde.pde_dataset(conf_data)
trainX, trainY, coordinates, test_set= data_loader.load("WakeCylinder_train.mat", "WakeCylinder_test.mat") # = trainX, testX, trainY, testY, vmin, vmax
coordinates_small = genfromtxt("vertices_small.csv", delimiter=",")
mynet = due.networks.transformer.pit(coordinates, coordinates_small, conf_train["device"], conf_net)
#mynet = mynet.load_params(conf_train["save_path"]+"/model")
model = due.models.pde.PDE(trainX, trainY, mynet, conf_train)
model.train()
model.save_hist()

pred = mynet.predict(test_set[...,:conf_data["memory"]+1], 10, device=conf_train["device"])
elements        = genfromtxt("elements.csv", delimiter=",").astype("int32")-1
due.utils.pde2dirregular_evaluate(coordinates, elements, prediction=pred, truth=test_set, save_path=conf_train["save_path"])