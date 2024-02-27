import pickle
import csv
import numpy as np
import pandas as pd
import math
from glob import glob
from matplotlib import pyplot  as plt
from scipy.special import erfcinv
import scipy.optimize 
import random 
from scipy import interpolate
curve_fit = scipy.optimize.curve_fit
import matplotlib.cm as cm
import copy
from sys import argv
import os
from Charge_dist_DEggPMT_for_uniformity_data_newtheta import Charge_dist_DEggPMT





def get_effective_area(df, r_range, theta_range):
    # Ngen = 270.2282445465208 ##DEgg2021-3-135
    # df = df.to_numpy() / Ngen
    theta_range = theta_range * np.pi / 180
    dth = theta_range[1]-theta_range[0]
    phi2, r2 = np.meshgrid(theta_range, r_range, indexing="ij")
    dS = ((r2[:,1:]/10)**2-(r2[:,:-1]/10)**2)*dth/2

    dAeff = dS*(df[:,1:]+df[:,:-1])/2
    print("Effective area = ", np.sum(dAeff))

    Area_check = np.sum(dS[r2[:,1:]<120])
    Area_theory = np.pi*(110/10)**2
    print("check=", Area_check)
    print("thero=", Area_theory)  
    # else:
    #     print("No effective area")



chgdist = Charge_dist_DEggPMT(775, isDebug = False)

rhoRange = np.arange(0, 141, 3)
phiRange = np.arange(0, 360, 6)
Rho, Phi = np.meshgrid(rhoRange, phiRange)

hitmap = np.zeros((len(phiRange), len(rhoRange)), dtype=np.float32)
for nphi, phi in enumerate(phiRange):
    for nrho, rho in enumerate(rhoRange):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        eff = chgdist.GetChargeDistFromLocalXY(x, y)
        np.add.at(hitmap, (nphi, nrho), eff)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.pcolormesh(Phi, Rho, hitmap)
plt.show()


get_effective_area(hitmap, rhoRange, phiRange)