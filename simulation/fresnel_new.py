import matplotlib.pyplot as plt
import math
import numpy as np



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
    

alpha = np.linspace(0, 90, 50)
alpha = np.deg2rad(alpha)
print(alpha)
n1 = 1.43
n2 = 1.5
# for i in alpha:
#     # print(math.sin(math.radians(i)))
#     sin_beta = math.sin(math.radians(i)) / n2 
#     # print(sin_beta)
#     beta.append(math.degrees(math.asin(sin_beta)))  
ts = 2 * n1 * np.cos(alpha) / (n1 * np.cos(alpha) + n2 * np.sqrt(1 - (n1 / n2 * np.sin(alpha)) ** 2))
tp = 2 * n1 * np.cos(alpha) / (n2 * np.cos(alpha) + n1 * np.sqrt(1 - (n1 / n2 * np.sin(alpha)) ** 2))
# print(ts)
# print(tp)
Ts = np.absolute(ts) ** 2
Tp = np.absolute(tp) ** 2
T = (Ts + Tp) / 2

Rs = 1 - Ts
Rp = 1 - Tp
R = (Rs + Rp) / 2 


plt.figure()
plt.grid()
# plt.plot(alpha, Rs_list, label = 's')
# plt.plot(alpha, Rp_list, label = 'p')
plt.title(f"fresnel's equation n = {n2/n1:.2f}", fontsize = 20)
plt.ylabel('T, R', fontsize = 18)
plt.xlabel('degree', fontsize = 18)
# plt.plot(np.degrees(alpha), Ts, label = 's')
# plt.plot(np.degrees(alpha), Tp, label = 'p')
plt.plot(np.degrees(alpha), get_transmittance(alpha), label = 'T')
# plt.plot(alpha, T_list)
# plt.plot(alpha, Rp_list)
plt.plot(np.degrees(alpha), 1 - get_transmittance(alpha), label = "R")
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.tight_layout()
plt.savefig("./frenel.png")
plt.show()
