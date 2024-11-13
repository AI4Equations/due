from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

# Load the data
# data = loadmat("../DampedPendulum_partial/DampedPendulum_partial_train.mat")["trajectories"]
# print(data.shape)
# # Add noise to the data
# data *= ( 1 + np.random.uniform(-0.05, 0.05, data.shape) )
# savemat("DampedPendulum_partial_train.mat", mdict={"trajectories": data})

# plt.plot(data[0,0,:])
# plt.savefig("DampedPendulum_partial_train_noisy.pdf", dpi=300)

t = np.linspace(0, 20, 1001).reshape(-1,1)
true = loadmat("DampedPendulum_partial_test.mat")["trajectories"][0,...].transpose()
pred5 = loadmat("model_0.05/pred.mat")["trajectories"][0,...].transpose()
pred10 = loadmat("model_0.1/pred.mat")["trajectories"][0,...].transpose()

np.savetxt("true_1_3.csv", np.hstack([t,true, pred5, pred10]), delimiter=" ")

# plt.plot(true[0,0,:], label="True")
# plt.plot(pred5[0,0,:], label="Pred5")
# plt.plot(pred10[0,0,:], label="Pred10")
# plt.legend()
# plt.savefig("pred.pdf", dpi=300)