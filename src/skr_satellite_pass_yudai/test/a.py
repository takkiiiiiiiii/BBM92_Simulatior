from os import getcwd
from os.path import join, basename
from sys import path 

libs_dir = join("/".join(getcwd().split("/")[:-3]))
print(libs_dir)
path.append(libs_dir)


import numpy as np
from libs.qchannel_model import *
import matplotlib.pyplot as plt
from libs.figure_config import *
from libs.default_parameters import *
from libs.simulation_tools import *
from qiskit_aer import AerSimulator
from libs.satellite import *


insta_eta_alice = 0.00084288
insta_eta_bob = 0.00237062
num_detected_event = 0

yield_val = yield_from_photon_number(1, p_dark, insta_eta_alice, insta_eta_bob)
# print(f'Yield value: {yield_val}')

# Store photon number probabilities in a dictionary with keys like 'p_0', 'p_1', ...
photon_probs = {}
for i in range(10):
    photon_probs[f'p_{i}'] = photon_number_probability(i, lambda_signal)

# 0~9までの確率をリストにまとめる
prob_list = [photon_probs[f'p_{i}'] for i in range(10)]
# 残りの確率をp_10として補完
p_10 = 1.0 - sum(prob_list)
prob_list.append(p_10)

# 0~10のフォトン数に応じて10**6個のサンプルを生成
photon_list = np.random.choice(range(11), size=10**6, p=prob_list)
unique, counts = np.unique(photon_list, return_counts=True)
photon_counts = dict(zip(unique, counts))

for i in range(11):
    int(photon_counts.get(np.int64(i), 0))
    print(f'Photon number {i}: {int(photon_counts.get(np.int64(i), 0))}')
    

        
