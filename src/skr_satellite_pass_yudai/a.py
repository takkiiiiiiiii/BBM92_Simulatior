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


insta_eta_alice = 0.00084288
insta_eta_bob = 0.00237062
num_detected_event = 0

yield_val = yield_from_photon_number(1, Y0_A, Y0_B, insta_eta_alice, insta_eta_bob)
# print(f'Yield value: {yield_val}')

for i in range(10**6):
    # Simulate a random number to determine if an event is detected
    # This is a placeholder for the actual detection logic
    # In practice, this would be based on the photon counts and other parameters
    random_number = np.random.rand()
    # print(f'Random number: {random_number}')
    if random_number < yield_val :
        num_detected_event += 1
        
print(f'Number of detected events: {num_detected_event}')