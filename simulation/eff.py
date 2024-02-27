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

def main():
    measurement_type = input("measurement type: ")
    condition = input("condition: ")
    fresnel = input("theta or transmittance?: ")
    data_dir = f"./{measurement_type}/{condition}/"
    HitposList = []
    Ts = np.array([])
    thetas = np.array([])
    files = glob(f"{data_dir}top*")
    for file in tqdm(files):
        df = ur.open(file)
        tree = df["tree"]
        hitpos = tree["LocalHitPos"].array(library = "np")
        HitposList.append(hitpos)
        mom = tree["LocalDir"].array(library = "np")
        theta = get_theta(hitpos, mom)
        T = get_transmittance(theta)
        Ts = np.append(Ts, T)
        thetas = np.append(thetas, theta)
        # Ts.append(T)
    hitposs = np.vstack(HitposList)
    Ts = np.ravel(Ts)
    print("measurement type: ", measurement_type)
    thetas = np.ravel(thetas)

    plt.title(f"{measurement_type}", fontsize = 25)
    plt.yscale("log")
    plt.ylabel("events", fontsize = 20)
    # plt.hist(np.degrees(thetas), bins = 50)
    if fresnel == "theta":
        plt.hist(thetas, bins = 50)
    elif fresnel == "transmittance":
        plt.hist(Ts, bins = 50)
    plt.xlabel(f"{fresnel}", fontsize = 20)
    # plt.xlabel("degree", fontsize = 20)
    # plt.tight_layout()
    # plt.savefig(f"./figs/theta_hist_{measurement_type}scan.png")
    plt.savefig(f"./figs/{fresnel}_hist_{measurement_type}_{condition}.pdf")
    # plt.show()
    plt.close()


    nbin = 2500
    x = hitposs[:, 0][0:-1:nbin]
    y = hitposs[:, 1][0:-1:nbin]
    z = hitposs[:, 2][0:-1:nbin]
    thetas = np.degrees(thetas[0:-1:nbin])
    Ts = Ts[0:-1:nbin]
    X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), len(x)), np.linspace(np.min(y), np.max(y), len(y)))
    if fresnel == "theta":
        nz = griddata((x, y), thetas, (X, Y))
    elif fresnel == "transmittance":
        nz = griddata((x, y), Ts, (X, Y))

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(X, Y, nz, color = "jet")
    ax.set_title(f"{measurement_type}", fontsize = 25)
    ax.set_xlabel("X [mm]", fontsize = 20)
    ax.set_ylabel("Y [mm]", fontsize = 20)
    clb = fig.colorbar(mesh, orientation = "vertical")
    clb.set_label(f"{fresnel}", fontsize = 20)
    plt.tight_layout()
    plt.savefig(f"./figs/{fresnel}_2dplot_{measurement_type}_{condition}.pdf")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
