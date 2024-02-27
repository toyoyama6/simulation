import numpy as np
from scipy import interpolate 
# import scipy.interpolate as spi
import matplotlib.pyplot as plt
import pandas as pd



dis = [7, 20]



df = pd.read_hdf("./theta_pde.hdf5")
dfnp = df.to_numpy()
boundary = 59.49728221842466



plt.plot(np.linspace(0, 60, 64), np.mean(dfnp, axis = 0), label = "0 mm", linewidth = 5, color = "black")
for j in dis:
    theta = np.load(f"./{j}mm_theta.npy")
    newtheta = []
    for i in range(len(dfnp)):    
        # x = np.append(theta, boundary)
        # y = np.append(dfnp[i, :], 0)
        x = theta
        y = dfnp[i,:]
        f = interpolate.interp1d(x, y, kind = "slinear", fill_value = "extrapolate")
        xd = np.linspace(np.min(x), boundary, 64)
        newtheta.append(f(xd))
        # plt.plot(xd, f(xd))
        # print(f(xd))
    # print(np.shape(newtheta))
    plt.plot(xd, np.mean(newtheta, axis = 0), label = f"{j} mm")#np.mean(f(xd)), axis = 0)
        # plt.show()
    newtheta = pd.DataFrame(newtheta, columns = xd)
    newtheta.to_hdf(f"./{j}mm_interpolate1.hdf", key = "newtheta")
    # for i in range(len(dfnp)):    
    #     f = interpolate.interp1d(theta2, dfnp[i, :], kind = "linear", fill_value = "extrapolate")#, fill_value="extrpolate")
    #     y = f(x)
    # np.save(f"./{j}mm_interpolate.npy", np.array(newtheta))

    # plt.plot(theta, dfnp)
    # plt.plot(x, y)
    # plt.show()
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel(r"$\theta$", fontsize = 20)
plt.ylabel("Efficiency", fontsize = 20)
plt.legend(fontsize = 18)
plt.tight_layout()
plt.savefig("./NewThetaDistribution.pdf")
plt.show()
