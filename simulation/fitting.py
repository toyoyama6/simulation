import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm


def fitting_function(x, A, k, x0, y0):
    return -A / (1 + np.exp(-k * (x - x0))) + y0


def get_parameter(x, y, p0):
    popt, pcov = curve_fit(fitting_function, x, y, p0=p0, maxfev=10000000)
    return popt


df = pd.read_hdf("./theta_pde.hdf5")

A_var = np.linspace(-0.1, 0.1, len(df))
k_var = np.linspace(-0.1, 0.1, len(df))
x0_var = np.linspace(-8, 8, len(df))
y0_var = np.linspace(-0.02, 0.02, len(df))

x = np.linspace(-90, 90, 100)
fig, ax = plt.subplots(figsize = (12,8))
for i in range(len(df)):
    p0 = [0.8, 0.2, 60, 0.93]
    popt = get_parameter(df.columns.values, df.iloc[i, :], p0)
    print(popt)
    ax.plot(x, fitting_function(x, popt[0], popt[1], popt[2], popt[3]), color="green", label="my function")
    # ax.plot(x, fitting_function(x, 0.8, 0.2, 60, 0.93), color="black", label="my function")
    # ax.plot(df.columns.values, df.iloc[i, :], color="red", label="uniformity data")
# plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.ylim(0,1)
plt.legend(
    by_label.values(),
    by_label.keys(),
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0,
    fontsize=15,
)
ax.tick_params("both", labelsize = 15)
ax.set_xlabel(r"$\theta$", fontsize = 20)
fig.tight_layout()
plt.savefig("./uniformity.pdf")
plt.show()