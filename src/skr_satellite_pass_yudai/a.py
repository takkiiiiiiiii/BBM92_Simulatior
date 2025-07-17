from os import getcwd
from os.path import join, basename
from sys import path 

libs_dir = join("/".join(getcwd().split("/")[:-2]))
print(libs_dir)
path.append(libs_dir)

# filename = basename(globals()['__vsc_ipynb_file__']).split(".")[0]

import numpy as np
from libs.qchannel_model import *
import matplotlib.pyplot as plt
from libs.figure_config import *
from libs.default_parameters import *
from libs.simulation_tools import *
from qiskit_aer import AerSimulator
from libs.satellite import *


satellite_1293 = LEOsatellite(
    libs_dir + '/data/STARLINK_1293.txt')
location_aizu = np.loadtxt(
    libs_dir + '/data/ogs_loc.txt')
location_sendai = np.loadtxt(
    libs_dir + '/data/ogs_loc_sendai.txt')
year = 2021
day = 357
hour = 16
minute = 29

t = np.arange(0, 241, 10) 
second = 56 + t 
utc = 9

latitude_bob = location_aizu[:, 0]
longitude_bob = location_aizu[:, 1]
elevation_bob = location_aizu[:, 2]

latitude_alice = location_sendai[:, 0]
longitude_alice = location_sendai[:, 1]
elevation_alice = location_sendai[:, 2]

# bob用のリスト
slant_path_bob_lst = np.zeros(len(second))
zenith_angle_bob_lst = np.zeros(len(second))

# alice用のリスト
slant_path_alice_lst = np.zeros(len(second))
zenith_angle_alice_lst = np.zeros(len(second))

for idx in range(len(second)):
    # bobとの幾何学的関係を計算
    slant_path_bob_lst[idx], zenith_angle_bob_lst[idx] = satellite_1293.computeGeometricWithUser(
        year, day, hour, minute, second[idx], utc,
        longitude_bob[0], latitude_bob[0], elevation_bob[0] # bobが単一の場合、[0]でアクセス
    )

    slant_path_alice_lst[idx], zenith_angle_alice_lst[idx] = satellite_1293.computeGeometricWithUser(
        year, day, hour, minute, second[idx], utc,
        longitude_alice[0], latitude_alice[0], elevation_alice[0]
    )

slant_path_bob_lst = slant_path_bob_lst * 1000
slant_path_alice_lst = slant_path_alice_lst * 1000
print(f'zenith_angle_alice_deg: {zenith_angle_alice_lst}')
print(f'zenith_angle_bob_deg: {zenith_angle_bob_lst}')

qber_measured = [
    0.05382865, 0.05319561, 0.05177840, 0.05354289,
    0.05204841, 0.05238488, 0.05205546, 0.05225517,
    0.05265290, 0.05231749, 0.05277432, 0.05199989,
    0.05166382, 0.05279676, 0.05170554, 0.05237383
]
qber_list = [
    0.05345017, 0.05319561, 0.05296401, 0.05275616, 0.05256434, 0.05238488,
    0.05221606, 0.05205857, 0.05191608, 0.05179595, 0.05170698, 0.05165837,
    0.05165597, 0.05170009, 0.05178532, 0.05140277, 0.05194315, 0.05239920,
    0.05296681, 0.05154474, 0.05273519, 0.05214094, 0.05300032, 0.05342061,
    0.05411128
]

qber_simulated_tail = qber_list[-9:]

# 合成して qber_simul に
qber_simul = qber_measured + qber_simulated_tail
print(f'qber_simul: {qber_simul}')


plt.scatter(t, [q * 100 for q in qber_simul])
plt.xlabel(r"Satellite Pass Duration [s]")
plt.ylabel("QBER [%]")
plt.grid()
plt.tight_layout()
# plt.show()


print(len(qber_simul))
np.save('results/qber_simul_bbm92_test1_n_s1_new_081.npy', qber_simul, allow_pickle=True)
