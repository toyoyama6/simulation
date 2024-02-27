import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_hdf("./2d_data.hdf")

r = df.columns.values * 3
theta = df.index.values * 6 * np.pi / 180

R, THETA = np.meshgrid(r, theta)

X = R * np.cos(THETA) - 16
Y = R * np.sin(THETA) - 9.6
plt.pcolormesh(X, Y, df)
plt.grid()
plt.xlabel("x [mm]", fontsize = 25)
plt.ylabel("y [mm]", fontsize = 25)
plt.tight_layout()
plt.savefig("./xyscan.png")
plt.show()
plt.close()

plt.plot(r, df.T)
plt.show()
plt.close()

print(theta)
print(len(df)/2)
for i in range(int(len(df) / 2)):
    plt.plot(r, df.iloc[i,:], color = "red")
    plt.plot(-r, df.iloc[int(len(df) / 2) + i, :], color = "blue")  
plt.show()
