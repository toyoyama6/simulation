import numpy as np
import matplotlib.pyplot as plt
import uproot as ur
from scipy.interpolate import griddata
from tqdm import tqdm
from glob import glob


R1 = 131.0
R2 = 43.9106
R3 = 57.0894
z0 = 59.5

def sphere1(x, y, z):
    return x ** 2 + y ** 2 + (z - 59.5) ** 2 - R1 ** 2

def diff_sphere1(x, y, z):
    return np.array([2 * x, 2 * y, 2 * (z - z0)])
    
def circle(x, y, z):
    a = np.sqrt(R2 ** 2 / (x ** 2 + y ** 2))
    return (x - a * x) ** 2 + (y - a * y) ** 2 + (z - 59.5) ** 2 - R3 ** 2

def diff_circle(x, y, z):
    a = np.sqrt(R2 ** 2 / (x ** 2 + y ** 2))
    return np.array([2 * (1 - a) ** 2 * x, 2 * (1 - a) ** 2 * y, 2 * (z - z0)])

def get_theta(hitpos, mom):
    rho = np.sqrt(hitpos[:, 0] ** 2 + hitpos[:, 1] ** 2) 
    surface_vector = np.where(rho > 101, -1, np.where(rho < 80, diff_sphere1(hitpos[:, 0 ], hitpos[:,1], hitpos[:,2]), diff_circle(hitpos[:, 0], hitpos[:, 1], hitpos[:, 2]))).T
    abs_surface_vector = np.linalg.norm(surface_vector, axis = 1) 
    norm_vector = surface_vector / abs_surface_vector[:, None]
    abs_norm_vector = np.linalg.norm(norm_vector, axis=1)
    abs_mom = np.linalg.norm(mom, axis = 1)
    inner = np.array(norm_vector[:,0] * mom[:,0] + norm_vector[:,1] * mom[:,1] + norm_vector[:,2] * mom[:,2])
    cos = np.absolute(inner / abs_norm_vector / abs_mom)
    theta = np.arccos(cos)
    return theta

"""    
    rho = np.sqrt(x ** 2 + y ** 2) 
    if rho < 80:
        surface_vector = diff_sphere1(x, y, z)
    elif 80 < rho < 101:
        surface_vector = diff_circle(x, y, z)
    norm_vector = surface_vector / np.linalg.norm(surface_vector) 
    abs_norm_vector = np.linalg.norm(norm_vector)
    abs_mom = np.linalg.norm(mom)
    inner = np.inner(norm_vector, mom)
    cos = np.absolute(inner / abs_norm_vector / abs_mom)
    # if cos < 0:
    #     cos = -cos
    try:
        theta = np.arccos(cos)
    except:
        theta = -1
    return theta
"""



def get_transmittance(theta):
    n1 = 1.43 # gel
    n2 = 1.5 # PMT glass
    ts = 2 * n1 * np.cos(theta) / (n1 * np.cos(theta) + n2 * np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2))
    tp = 2 * n1 * np.cos(theta) / (n2 * np.cos(theta) + n1 * np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2))
    Ts = np.absolute(ts) ** 2
    Tp = np.absolute(tp) ** 2
    T = (Ts + Tp) / 2
    # Rs = 1 - Ts
    # Rp = 1 - Tp
    # R = (Rs + Rp) / 2 
    return T

# x = np.linspace(0, 90, 100)
# plt.plot(np.deg2radx, get_transmittance(x))
# plt.show()

# mom = np.array([-0.00156632, -0.00109584, -0.99999815], dtype = np.float32)
# hitpos = np.array([0.7073146 , 0.49485555, -130.99716], dtype = np.float32)
# theta = get_theta(hitpos[0], hitpos[1], hitpos[2], mom)
# print(theta)


Ts = []
files = glob("./top*")
for file in tqdm(files):
    df = ur.open(file)
    tree = df["tree"]
    hitpos = tree["LocalHitPos"].array(library = "np")
    mom = tree["LocalDir"].array(library = "np")
    theta = get_theta(hitpos, mom)
    T = get_transmittance(theta)
    plt.hist(np.degrees(theta))
    # Ts.append(T)
# mom = np.array([-0.00156632, -0.00109584, -0.99999815], dtype = np.float32)
# hitpos = np.array([0.7073146 , 0.49485555, -130.99716], dtype = np.float32)]

# plt.yscale("log")
# plt.hist(np.degrees(get_theta(hitpos, mom)))
# plt.show()
plt.savefig("./theta.png")
plt.show()
"""

theta_list = []
for i in hitpos:
    # x = hitpos[i][0]
    # y = hitpos[i][1]
    # z = hitpos[i][2]
    theta = get_theta(i, mom)
    theta_list = np.append(theta_list, theta)

plt.hist(theta_list)
plt.show()

x = hitpos[:, 0]
y = hitpos[:, 1]
z = hitpos[:, 2]

T = get_transmittance(theta_list)
plt.hist(T, bins = 100)
plt.xscale("log")
plt.show()
X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), len(x)), np.linspace(np.min(y), np.max(y), len(y)))

nz = griddata((x, y), T, (X, Y))
# plt.scatter(, hitpos[:, 1])
# plt.show()
# plt.close()

plt.contourf(X, Y, nz)
plt.colorbar()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection' : '3d'})
ax.scatter(hitpos[:, 0], hitpos[:, 1], hitpos[:, 2])
plt.show()
surface_vector = diff_sphere1(hitpos[0], hitpos[1], hitpos[2])
norm_vector = surface_vector / np.linalg.norm(surface_vector) 
abs_norm_vector = np.linalg.norm(norm_vector)
abs_mom = np.linalg.norm(mom)



fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection' : '3d'})
# ax.scatter(surface_vector[0] + norm_vector[0], surface_vector[1] + norm_vector[1], surface_vector[2] + norm_vector[2], color = "green")
# ax.scatter(hitpos[0] + surface_vector[0], hitpos[1] + surface_vector[1], hitpos[2] + surface_vector[2], color = "black")
# ax.scatter(hitpos[0], hitpos[1], hitpos[2], color = "green")
ax.quiver(hitpos[0], hitpos[1], hitpos[2],
            hitpos[0] + norm_vector[0], hitpos[1] + norm_vector[1], hitpos[2] + norm_vector[2],
            color = "blue", length=1, arrow_length_ratio=0.3, pivot='tail', normalize=False)
# ax.quiver(hitpos, hitpos + mom, color = "black")
ax.set_xlim(0.69, 1)
ax.set_ylim(0.49, 0.7)
ax.set_zlim(-190, -129)
plt.show()

"""