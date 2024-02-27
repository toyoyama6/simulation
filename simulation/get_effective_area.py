import h5py 
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  
import warnings
warnings.filterwarnings("ignore")
import glob 
import sys 


Ngen = 421.0
outfh = h5py.File("2d_data.hdf", 'r')
r = outfh["df"]["axis0"][()]*3.0
phi = outfh["df"]["axis1"][()]*6.0*np.pi/180
outfh["df"]["block0_items"][()]
z = outfh["df"]["block0_values"][()]/1.6
zmax = np.max(z)
# z [phi, r]
for z_r in z:
    #plt.plot((r[1:]+r[:-1])/2, (z_r[1:]+z_r[:-1])/2/Ngen, "-", alpha=0.2)
    plt.plot(r, z_r, "-", alpha=0.2)
    
plt.show()
dth = phi[1]-phi[0]
phi2, r2 = np.meshgrid(phi, r, indexing="ij")
dS = (r2[:,1:]**2-r2[:,:-1]**2)*dth/2
dAeff = dS*(z[:,1:]+z[:,:-1])/2/Ngen
print("Effective area = ", np.sum(dAeff))
Area_check = np.sum(dS[r2[:,1:]<110])
Area_theory = np.pi*110**2
print("check=", Area_check)
print("thero=", Area_theory)
