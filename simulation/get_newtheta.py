import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


R1 = 131
R2 = 57.0894
rc = 43.9106
z0 = 59.5


def get_theta_sphere(theta, meas_err):
    theta = theta * np.pi / 180
    lamda = (-2 * meas_err * np.cos(theta) + np.sqrt((2 * meas_err * np.cos(theta)) ** 2 - 4 * (meas_err ** 2 - R1 ** 2))) / 2
    z = meas_err + lamda * np.cos(theta)
    r = R1
    theta = np.arccos(z / r) * 180 / np.pi
    return theta

def get_theta_donut(theta, meas_err):
    theta = theta * np.pi / 180
    lamda = (-2 * (meas_err * np.cos(theta) - rc * np.sin(theta) - z0 * np.cos(theta)) + np.sqrt(4 * (meas_err * np.cos(theta) - rc * np.sin(theta) - z0 * np.cos(theta)) ** 2 - 4 * (rc ** 2 + meas_err ** 2 + z0 ** 2 - 2 * meas_err * z0 - R2 ** 2))) / 2
    r = np.sqrt(lamda ** 2 * np.sin(theta) ** 2 + (meas_err + lamda * np.cos(theta)) ** 2)
    z = meas_err + lamda * np.cos(theta)
    theta = np.arccos(z / r) * 180 / np.pi
    return theta


def get_newtheta(theta, meas_err):
    theta_boudary = np.arctan(80 / 103.7352) * 180 / np.pi
    new_theta = np.where(theta < theta_boudary, get_theta_sphere(theta, meas_err), get_theta_donut(theta, meas_err))
    # if theta < theta_boudary:
    #     new_theta = get_theta_sphere(theta, meas_err)
    # else:
    #     new_theta = get_theta_donut(theta, meas_err)
    return new_theta

def plot_eff(df, df_myfunc, new_theta_8mm, new_theta_12mm, new_theta_16mm, new_theta_18mm,new_theta_20mm,  theta):
    plt.title("AE distribution", fontsize = 25)
    plt.xlabel(r"$\theta$", fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.plot(theta, np.mean(df.T, axis = 1), label = "morii")
    plt.plot(new_theta_8mm, np.mean(df.T, axis = 1), label = "8mm")
    plt.plot(new_theta_12mm, np.mean(df.T, axis = 1), label = "12mm")
    plt.plot(new_theta_16mm, np.mean(df.T, axis = 1), label = "16mm")
    plt.plot(new_theta_18mm, np.mean(df.T, axis = 1), label = "18mm")
    plt.plot(new_theta_20mm, np.mean(df.T, axis = 1), label = "20mm")
    # plt.plot(theta, df_myfunc, label = "logis")
    #clb = plt.colorbar(cc)
    #clb.ax.tick_params(labelsize = 15)
    plt.legend(fontsize = 18)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"./new_uniformity.pdf")
    plt.close()

def logistic_function(x, A, k, x0):
    return -1 / (1 + A * np.exp(-k * (x - x0))) + 1

def main():
    theta = np.linspace(0,60,64)
    phi = np.linspace(0,360,72)
    df = pd.read_hdf("./theta_pde.hdf5")    
    p0 = [1000, 0.3, 30]
    dfnp = df.to_numpy()
    
    df_myfunc = np.mean(dfnp, axis = 0)
    df_myfunc = df_myfunc.T * logistic_function(theta, p0[0], p0[1], p0[2]).T
    df_myfunc = df_myfunc.T
    meas_errList = [-8, 7, 8, 12, 16, 17, 17.5, 18, 19, 20]
    # meas_err1 = 8 ##mm
    # meas_err2 = 12 ##mm
    # meas_err3 = 16 ##mm
    # meas_err4 = 18 ##mm
    # meas_err5 = 20 ##mm
    # new_theta_8mm = get_newtheta(theta, meas_err1)
    # new_theta_12mm = get_newtheta(theta, meas_err2)
    # new_theta_16mm = get_newtheta(theta, meas_err3)
    # new_theta_18mm = get_newtheta(theta, meas_err4)
    # new_theta_20mm = get_newtheta(theta, meas_err5)
    for meas_err in meas_errList:
        NewTheta = get_newtheta(theta, meas_err)
        np.save(f"./{meas_err}mm_theta.npy", NewTheta)
        plt.plot(NewTheta, np.mean(df.T, axis = 1), label = f"{meas_err} mm")    
    plt.title("AE distribution", fontsize = 25)
    plt.xlabel(r"$\theta$", fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    # plt.plot(theta, np.mean(df.T, axis = 1), label = "morii")
    # plt.plot(new_theta_8mm, np.mean(df.T, axis = 1), label = "8mm")
    # plt.plot(new_theta_12mm, np.mean(df.T, axis = 1), label = "12mm")
    # plt.plot(new_theta_16mm, np.mean(df.T, axis = 1), label = "16mm")
    # plt.plot(new_theta_18mm, np.mean(df.T, axis = 1), label = "18mm")
    # plt.plot(new_theta_20mm, np.mean(df.T, axis = 1), label = "20mm")
    # plt.plot(theta, df_myfunc, label = "logis")
    #clb = plt.colorbar(cc)
    #clb.ax.tick_params(labelsize = 15)
    plt.legend(fontsize = 18)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"./new_uniformity.pdf")
    plt.close()

    # np.save("./8mm_theta.npy", new_theta_8mm)
    # np.save("./12mm_theta.npy", new_theta_12mm)
    # np.save("./16mm_theta.npy", new_theta_16mm)
    # np.save("./18mm_theta.npy", new_theta_18mm)
    # np.save("./20mm_theta.npy", new_theta_20mm)
    # plot_eff(dfnp, df_myfunc, new_theta_8mm, new_theta_12mm, new_theta_16mm, new_theta_18mm, new_theta_20mm, theta)

if __name__=="__main__":
    main()