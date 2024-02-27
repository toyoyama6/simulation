import matplotlib.pyplot as plt
import math
import numpy as np


alpha = np.linspace(0, 45, 50)
# alpha = np.deg2rad(alpha)
beta = []
Rs_list = []
Rp_list = []
Ts_list = []
Tp_list = []
T_list = []
n1 = 1
n2 = 1.46
# for i in alpha:
#     # print(math.sin(math.radians(i)))
#     sin_beta = math.sin(math.radians(i)) / n2 
#     # print(sin_beta)
#     beta.append(math.degrees(math.asin(sin_beta)))  
for i in alpha:
    cos = math.cos(math.radians(i))
    sin = math.sin(math.radians(i))
    x = math.sqrt(n2 ** 2 - sin ** 2) 
    y = math.sqrt(1 - (n1 / n2) ** 2 * sin ** 2)
    rs = (cos - x) / (cos + x)
    Rs = rs ** 2
    Ts = 1 - Rs
    Rs_list.append(Rs)
    Ts_list.append(Ts) 
    rp = (n2 * cos - n1 * y) / (n2 * cos + n1 * y)
    Rp = rp ** 2
    Tp = 1 - Rp
    Rp_list.append(Rp)
    Tp_list.append(Tp)
    T = (Ts + Tp) / 2
    # print(T)
    T_list.append(T)


print(alpha)
print(T_list)
plt.figure()
plt.grid()
# plt.plot(alpha, Rs_list, label = 's')
# plt.plot(alpha, Rp_list, label = 'p')
plt.title('transmittance n = 1.46', fontsize = 20)
plt.ylabel('T', fontsize = 18)
plt.xlabel('degree', fontsize = 18)
plt.plot(alpha, Ts_list, label = 's')
plt.plot(alpha, Tp_list, label = 'p')
plt.plot(alpha, T_list)
# plt.plot(alpha, Rp_list)
# plt.plot(alpha, Rs_list)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend()
plt.show()
plt.savefig("./frenel.png")
