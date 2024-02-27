import numpy as np 
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


R1 = 131.0
R2 = 43.9106
R3 = 57.0894
z0 = 59.5 

def sphere1(x, y):
    return -np.sqrt(R1 ** 2 - x ** 2 - y ** 2)
    
def diff_sphere1(x, y, z):
    return np.array([2 * x, 2 * y, 2 * z])

def diff_circle(x, y, z):
    x0 = x / np.sqrt(x ** 2 + y ** 2) * R2
    y0 = y / np.sqrt(x ** 2 + y ** 2) * R2
    return np.array([2 * (x - x0), 2 * (y - y0), 2 * (z + z0)])

def circle(x, y):
    # a should be 59.5
    # a = np.sqrt(R2 ** 2 / (x ** 2 + y ** 2))
    x0 = x / np.sqrt(x ** 2 + y ** 2) * R2
    y0 = y / np.sqrt(x ** 2 + y ** 2) * R2
    return -np.sqrt(R3 ** 2 - (x - x0) ** 2 - (y - y0) ** 2) - z0 

def get_theta(hitpos, mom):
    x = hitpos[:, 0]
    y = hitpos[:, 1]
    z = hitpos[:, 2]
    plt.xlabel("y", fontsize = 20)
    plt.ylabel("z", fontsize = 20)
    plt.scatter(np.array(y), np.array(z),s = 0.1)# linestyle= '', marker='.')
    plt.tight_layout()
    plt.show()
    rho = np.sqrt(x ** 2 + y ** 2) 
    surface_vector = np.where(rho > 101, -1, np.where(rho < 80, diff_sphere1(x, y, z), diff_circle(x, y, z))).T ## surface vector
    norm_vector = surface_vector / np.linalg.norm(surface_vector, axis = 1)[:, None] ##nomilization
    inner = np.array(norm_vector[:,0] * mom[:,0] + norm_vector[:,1] * mom[:,1] + norm_vector[:,2] * mom[:,2]) ##calculate inner product
    cos = np.absolute(inner / np.linalg.norm(norm_vector, axis=1) / np.linalg.norm(mom, axis = 1))
    theta = np.arccos(cos) ## radian
    return theta, norm_vector

mom = np.array([0,0,1])
moms = np.tile(mom, (100000,1)) 
x = np.zeros(100000)
y = np.linspace(-101, 101, 100000)
rho = np.sqrt(x ** 2 + y ** 2)
z = np.where(rho > 101, -1, np.where(rho < 80, sphere1(x, y), circle(x, y)))
hitpos = np.stack([x,y,z], axis = 1)
theta, norm_vector = get_theta(hitpos, moms)

plt.scatter(y, theta, s = 0.1)
plt.xlabel("y", fontsize = 20)
plt.ylabel("theta", fontsize = 20)
plt.ylabel(r"$\theta$", fontsize = 20)
plt.show()




plt.ylim(-150, -50)
plt.plot(np.array(y), np.array(z), linestyle= '', marker='.')
for i in range(len(y)):
    # plt.quiver(y[i], z[i], moms[i,1], moms[i,2], width=0.01)
    plt.quiver(y[i], z[i], norm_vector[i,1], norm_vector[i,2], width=0.01)
plt.xlabel("y", fontsize = 20)
plt.ylabel("z", fontsize = 20)
plt.show()


x = np.linspace(-101, 101, 1000)
y = np.linspace(-101, 101, 1000)
z = np.where(rho > 101, -1, np.where(rho < 80, sphere1(x, y), circle(x, y)))

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
plt.plot(x, y, z, linestyle= '', marker='.')
ax.set_xlabel('x[mm]', fontsize = 20)
ax.set_ylabel('y[mm]', fontsize = 20)
ax.set_zlabel('z[mm]', fontsize = 20)
plt.tight_layout()
plt.show()
