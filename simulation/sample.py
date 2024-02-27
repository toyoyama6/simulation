import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot  as plt
from scipy.special import erfcinv
import scipy.optimize
curve_fit = scipy.optimize.curve_fit
import matplotlib.cm as cm
from sys import argv







def _convertXYtoZenPhi(X, Y):
    phi = np.arctan2(Y,X)*180.0/np.pi
    rho = np.sqrt(X*X+Y*Y)
    R1 = 131.0
    R2 = 57.0894
    zen = None
    if Y >= 0:
        zen = np.where(rho>101, -1, 
                        np.where(rho>80, np.arctan(rho / np.sqrt(R2 ** 2 - (X ** 2 + (Y-43.9106) ** 2))) * 180.0 / np.pi, 
                        np.arctan(rho / (np.sqrt(R1 ** 2 - rho ** 2) - 59.5)) * 180 / np.pi))
    elif Y < 0:
        zen = np.where(rho>101, -1, 
                        np.where(rho>80, np.arctan(rho / np.sqrt(R2 ** 2 - (X ** 2 + (Y+43.9106) ** 2))) * 180.0 / np.pi, 
                        np.arctan(rho / (np.sqrt(R1 ** 2 - rho ** 2) - 59.5)) * 180 / np.pi))

    return zen, phi


# zen, phi = _convertXYtoZenPhi(90, 0)
# print(zen, phi)



def _convertXYtoZenPhi_old(X, Y):
    phi = np.arctan2(Y,X)*180.0/np.pi
    rho = np.sqrt(X*X+Y*Y)
    R1 = 131.0
    R2 = 57.0894
    zen = None

    zen = np.where(rho>101, -1, np.where(rho>80, np.arctan(rho/(np.sqrt(R2*R2-(rho-43.9106)**2)+59.5))*180.0/np.pi, np.arcsin(rho/R1)*180.0/np.pi))
    return zen, phi

zen, phi = _convertXYtoZenPhi(90, 0)
print(zen, phi)

file = "theta_pde.hdf"
df = pd.read_hdf(file)