import due
####################################################
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.ode.ode_dataset_osg(conf_data)
trainX, trainY, test_traj, test_dt, vmin, vmax, tmin, tmax = data_loader.load("RobertsonOSG_train.mat", "RobertsonOSG_test.mat") # = trainX, trainY, test, vmin, vmax

mynet = due.networks.fcn.dual_osgnet(vmin, vmax, tmin, tmax, conf_net, multiscale=conf_data["multiscale"])

#### specify the dataset for gdsg method
####
import numpy as np
AC_rand = np.random.uniform(size=(trainX.shape[0]*conf_train["sg_pairing"], 2)).astype(conf_data["dtype"])
AC_rand = AC_rand / np.sum(AC_rand, axis=1, keepdims=True)
B_rand  = 5e-5 * np.random.uniform(size=(trainX.shape[0]*conf_train["sg_pairing"], 1)).astype(conf_data["dtype"])
u_rand  = np.concatenate((AC_rand[:,0:1], B_rand, AC_rand[:,1:2]),axis=1)
u_rand = u_rand / np.sum(u_rand, axis=1, keepdims=True)
print(u_rand[:,1].max(), np.sum(u_rand, axis=1))
u_rand = 2 * (u_rand - (vmax+vmin)/2) / (vmax-vmin)
u_rand = u_rand.reshape(trainX.shape[0], conf_train["sg_pairing"], trainY.shape[1])

dt_rand = np.random.uniform(low=tmin, high=tmax, size=(trainX.shape[0]*conf_train["sg_pairing"], 2)).astype(conf_data["dtype"])
if conf_data["multiscale"]:
    dt_rand = 10**dt_rand
dt12 = np.sum(dt_rand, axis=-1, keepdims=True)
if conf_data["multiscale"]:
    dt_rand = np.log10(dt_rand)
    dt12     = np.log10(dt12)
    
dt_rand = np.hstack((dt_rand,dt12))
dt_rand = 2 * (dt_rand - (tmax+tmin)/2) / (tmax-tmin)
dt_rand = dt_rand.reshape(trainX.shape[0], conf_train["sg_pairing"], 3)
osg_data = [u_rand, dt_rand]

model = due.models.ODE_osg(trainX, trainY, osg_data, mynet, conf_train)
model.train()
model.save_hist()


##################################
#mynet = mynet.load_params("model/model")
print(test_dt.shape)
test_dt = np.genfromtxt("full_dt_100000.csv").reshape(1,-1)
print(test_dt.shape)
pred = mynet.predict(test_traj[...,0], test_dt, device=conf_train["device"])
due.utils.ode_evaluate(prediction=pred, truth=None, save_path=conf_train["save_path"])

############
