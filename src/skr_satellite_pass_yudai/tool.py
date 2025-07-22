import random
import numpy as np
import math
from scipy.special import erf, erfc
from scipy.integrate import quad
from collections import Counter # For counting elements in the array

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

# --- Bell state generation, measurement, and sifting functions (unchanged) ---
def append_bell_state(qc, state_id, q0, q1):
    """Appends a specified Bell state to the quantum circuit."""
    if state_id == 1: # |Phi+> = (|00> + |11>) / sqrt(2)
        qc.h(q0)
        qc.cx(q0, q1)
    elif state_id == 2: # |Phi-> = (|00> - |11>) / sqrt(2)
        qc.h(q0)
        qc.cx(q0, q1)
        qc.z(q1)
    elif state_id == 3: # |Psi+> = (|01> + |10>) / sqrt(2)
        qc.h(q0)
        qc.cx(q0, q1)
        qc.x(q0)
    elif state_id == 4: # |Psi->: (|01> - |10>) / sqrt(2)
        qc.h(q0)
        qc.cx(q0, q1)
        qc.x(q0)
        qc.z(q1)

def generate_star_bell_circuit(num_pairs, selected_state_id):
    """Generates a quantum circuit with multiple Bell state pairs."""
    qc = QuantumCircuit(num_pairs * 2, num_pairs * 2) # num_qubits, num_clbits
    for i in range(num_pairs):
        q0 = i * 2 # Alice's qubit
        q1 = i * 2 + 1 # Bob's qubit
        append_bell_state(qc, selected_state_id, q0, q1)
    qc.barrier() # Separator for visualization
    return qc

def measure_star_users(qc, num_pairs, current_bob_id):
    """
    Measures Bell state pairs in random bases (Z or X) for Alice and Bob.
    Note: current_bob_id is not used for logic within this function.
    """
    alice_bases = []
    bob_bases = []

    for i in range(num_pairs):
        q0, q1 = i * 2, i * 2 + 1
        a_basis = random.choice(['Z', 'X']) # Alice's random basis choice
        b_basis = random.choice(['Z', 'X']) # Bob's random basis choice

        if a_basis == 'X':
            qc.h(q0) # Apply Hadamard for X-basis measurement
        if b_basis == 'X':
            qc.h(q1) # Apply Hadamard for X-basis measurement

        alice_bases.append(a_basis)
        bob_bases.append(b_basis)

    qc.barrier()
    qc.measure(range(qc.num_qubits), range(qc.num_qubits)) # Measure all qubits
    return alice_bases, bob_bases, [] # Return empty list for bob_assignment as it's not used

def sift_key_multi(alice_bits, bob_bits, alice_bases, bob_bases, selected_state_id):
    """
    Sifts the raw key bits based on matching bases and selected Bell state ID.
    Corrects Bob's bits if necessary based on the Bell state property.
    """
    sifted_alice = ''
    sifted_bob = ''
    for idx in range(len(alice_bits)):
        a_bit = alice_bits[idx]
        b_bit = bob_bits[idx]
        a_base = alice_bases[idx]
        b_base = bob_bases[idx]

        if a_base != b_base: # Only keep bits where bases match
            continue
        
        # Apply correction based on the selected Bell state and Bob's basis
        # This ensures Alice and Bob have correlated bits in the chosen basis
        flip = False
        if selected_state_id == 2 and b_base == 'X': # |Phi->: (00-11)/sqrt(2), X-basis measurement results are flipped
            flip = True
        elif selected_state_id == 3 and b_base == 'Z': # |Psi+>: (01+10)/sqrt(2), Z-basis measurement results are flipped
            flip = True
        elif selected_state_id == 4: # |Psi->: (01-10)/sqrt(2), Always flipped regardless of basis
            flip = True

        if flip:
            corrected_b_bit = '1' if b_bit == '0' else '0'
        else:
            corrected_b_bit = b_bit

        sifted_alice += a_bit
        sifted_bob += corrected_b_bit
    return sifted_alice, sifted_bob

# --- Helper function: Apply noise model ---
def apply_noise_model(p_alice_meas, p_bob_meas, num_pairs_for_circuit):
    """
    Creates a noise model with Pauli-X errors on measurement for Alice and Bob qubits.
    """
    noise_model = NoiseModel()
    
    # Define Pauli-X (bit-flip) error for Alice and Bob based on their individual QBERs
    error_alice = pauli_error([('X', p_alice_meas), ('I', 1 - p_alice_meas)])
    error_bob = pauli_error([('X', p_bob_meas), ('I', 1 - p_bob_meas)])
    
    # Apply measurement errors to Alice's qubits (even indices)
    for i in range(num_pairs_for_circuit):
        alice_qubit_idx = i * 2
        noise_model.add_quantum_error(error_alice, 'measure', [alice_qubit_idx])
        
    # Apply measurement errors to Bob's qubits (odd indices)
    for i in range(num_pairs_for_circuit):
        bob_qubit_idx = i * 2 + 1
        noise_model.add_quantum_error(error_bob, 'measure', [bob_qubit_idx])
        
    return noise_model

# --- Quantum Channel Model Functions ---
# Global parameters from default_parameters.py
# a = 0.75 # Aperture radius of the receiver telescope (m)
# n_s = 0.053 # Mean photon number per pulse (for single photon source)
# wavelength = 0.85e-6 # Wavelength of light (m)
# h_OGS = 10 # Height of Optical Ground Station (m)
# h_s = 20 * 1000 # Height of Satellite (m)
# h_atm = 20 * 1000 # Height of atmosphere (m)
# theta_rad = 0.2e-3 # Beam divergence angle (radians)
# v_wind = 21 # Wind speed (m/s) for Cn2 profile
# mu_x = 0 # Mean pointing error in x (m)
# mu_y = 0 # Mean pointing error in y (m)
# sigma_theta_x = theta_rad/8 # Standard deviation of pointing error in x (radians)
# sigma_theta_y = theta_rad/8 # Standard deviation of pointing error in y (radians)
# e_0 = 0.5 # Detector efficiency
# p_dark = 1e-5 # Dark count probability
# e_pol = 1/100 # Polarization misalignment error

a = 0.75 # Aperture radius of the receiver telescope (m)
n_s = 0.1 # Mean photon number per pulse (for single photon source)
wavelength = 0.85e-6 # Wavelength of light (m)
h_OGS = 10 # Height of Optical Ground Station (m)
h_s = 500 * 1000 # Height of Satellite (m)
h_atm = 20 * 1000 # Height of atmosphere (m)
theta_rad = 5e-6 # Beam divergence angle (radians)
v_wind = 21 # Wind speed (m/s) for Cn2 profile
mu_x = 0 # Mean pointing error in x (m)
mu_y = 0 # Mean pointing error in y (m)
sigma_theta_x = theta_rad/4 # Standard deviation of pointing error in x (radians)
sigma_theta_y = theta_rad/4 # Standard deviation of pointing error in y (radians)
e_0 = 0.5 # Detector efficiency
p_dark = 1e-5 # Dark count probability
e_pol = 1/100 # Polarization misalignment error


def compute_atm_loss(tau_zen, zenith_angle_rad):
    """Computes atmospheric loss based on zenith angle."""
    cos_val = np.cos(zenith_angle_rad)
    if cos_val == 0: # Avoid division by zero for straight overhead
        return 0.0
    tau_atm = tau_zen ** (1 / cos_val)
    if tau_atm > 1: # Ensure loss is not greater than 1 (i.e., transmittance not > 1)
        return 1.0
    return tau_atm

def rytov_variance(len_wave, zenith_angle_rad, h_OGS, h_atm, Cn2_profile):
    """Calculates Rytov variance for atmospheric turbulence."""
    k = 2 * np.pi / len_wave
    sec_zenith = 1 / np.cos(zenith_angle_rad)
    if np.cos(zenith_angle_rad) == 0:
        return float('inf')

    def integrand(h):
        return Cn2_profile(h) * (h - h_OGS)**(5/6)

    integral, _ = quad(integrand, h_OGS, h_atm)

    sigma_R_squared = 2.25 * (k)**(7/6) * sec_zenith**(11/6) * integral

    return sigma_R_squared

def Cn2_profile(h, v_wind=21, Cn2_0=1e-13):
    """Hufnagel-Valley model for refractive index structure parameter Cn2."""
    term1 = 0.00594 * (v_wind/27)**2 * (1e-5 * h)**10 * np.exp(-h/1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)

    return term1 + term2 + term3

def compute_slant_distance(h_s, H_g, zenith_angle_rad):
    """Computes the slant distance between satellite and ground station."""
    delta_h = h_s - H_g
    if np.cos(zenith_angle_rad) == 0:
        return float('inf')
    horizontal_distance = delta_h * np.tan(zenith_angle_rad)
    return np.sqrt(delta_h**2 + horizontal_distance**2)

def equivalent_beam_width_squared(a, w_L):
    """Calculates the equivalent beam width squared for pointing error."""
    if w_L == 0:
        return float('inf')
    nu = (np.sqrt(np.pi) * a) / (np.sqrt(2) * w_L)
    numerator = np.sqrt(np.pi) * erf(nu)
    denominator = 2 * nu * np.exp(-nu**2)
    if denominator == 0: # Avoid division by zero
        return w_L**2 * 1e10 # Return a very large value if denominator is zero
    return w_L**2 * (numerator / denominator)

def rvs_LN_fading(sigma_R_squared, size=1):
    """Generates random variables for log-normal fading."""
    if np.isinf(sigma_R_squared) or sigma_R_squared < 0:
        return np.array([0.0]) # No fading if variance is infinite or negative
    shape_param = np.sqrt(sigma_R_squared)
    I_a_random = np.random.lognormal(
        mean=-sigma_R_squared/2, sigma=shape_param, size=size)
    return I_a_random

def rvs_pointing_err(
        mu_x, mu_y, sigma_theta_x, sigma_theta_y,
        slant_distance, theta_rad, a, w_Leq_squared, size=1):
    """Generates random variables for pointing error attenuation."""
    if np.isinf(w_Leq_squared) or w_Leq_squared <= 0:
        return np.array([0.0])

    sigma_x_dist = sigma_theta_x * slant_distance
    sigma_y_dist = sigma_theta_y * slant_distance

    x = np.random.normal(loc=mu_x, scale=sigma_x_dist, size=size)
    y = np.random.normal(loc=mu_y, scale=sigma_y_dist, size=size)

    r = np.sqrt(x**2 + y**2)

    w_L_beam = slant_distance * theta_rad
    nu = (np.sqrt(np.pi) * a) / (np.sqrt(2) * w_L_beam)
    A0 = erf(nu)**2 # On-axis coupling efficiency
    eta_p = A0 * np.exp(-(2*r**2)/(w_Leq_squared)) # Pointing error attenuation

    return eta_p



# --- Optimized helper function for channel parameters ---
def _get_precalculated_channel_params(zenith_angle_deg, current_tau_zen):
    """Pre-calculates static channel parameters for efficiency."""
    zenith_angle_rad = np.deg2rad(zenith_angle_deg)
    
    slant_distance = compute_slant_distance(h_s, h_OGS, zenith_angle_rad)
    w_L = slant_distance * theta_rad

    sigma_R_squared = rytov_variance(
        wavelength, zenith_angle_rad, h_OGS, h_atm, Cn2_profile
    )
    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    eta_ell = compute_atm_loss(current_tau_zen, zenith_angle_rad)
    
    return {
        'slant_distance': slant_distance,
        'w_L_beam': w_L,
        'sigma_R_squared': sigma_R_squared,
        'w_Leq_squared': w_Leq_squared,
        'eta_ell': eta_ell
    }

# --- Optimized function to generate instantaneous transmittance ---
def generate_insta_eta_optimized(precalculated_params):
    """Generates instantaneous channel transmittance considering fading and pointing error."""
    slant_distance = precalculated_params['slant_distance']
    w_L_beam = precalculated_params['w_L_beam']
    sigma_R_squared = precalculated_params['sigma_R_squared']
    w_Leq_squared = precalculated_params['w_Leq_squared']
    eta_ell = precalculated_params['eta_ell']

    I_a = rvs_LN_fading(sigma_R_squared, size=1)[0] # Log-normal fading
    eta_p = rvs_pointing_err(
        mu_x, mu_y, sigma_theta_x, sigma_theta_y,
        slant_distance, theta_rad, a, w_Leq_squared, size=1
    )[0] # Pointing error attenuation
    
    insta_eta = eta_ell * I_a * eta_p # Total instantaneous transmittance
    return np.clip(insta_eta, 0, 1) # Clip to ensure valid transmittance (0 to 1)

# --- Binary Entropy Function H(x) ---
def binary_entropy(x):
    """Calculates the binary entropy H(x)."""
    x = np.clip(x, 1e-10, 1 - 1e-10) # Clip to avoid log(0) and log(1) issues
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

# --- NEW: P(n) function for photon number distribution ---
def P(n, λ):
    """
    Calculates the probability P(n) of having n photons in a pulse,
    based on the provided formula.
    """
    if n < 0:
        return 0.0
    numerator = (n + 1) * (λ ** n)
    denominator = (1 + λ) ** (n + 2)
    return numerator / denominator

# --- NEW: BBM92 Simulation Function ---
def bbm92simulation(num_pairs_for_qiskit_circuit, p_alice_meas, p_bob_meas, selected_state_id):
    """
    Simulates the BBM92 protocol for a given number of Bell state pairs and noise probabilities.
    This function encapsulates the Qiskit-specific simulation logic.

    Args:
        num_pairs_for_qiskit_circuit (int): The number of Bell state pairs to simulate in Qiskit.
                                            This is the number of 'effective' pairs detected.
        p_alice_meas (float): The individual QBER probability for Alice's measurement.
        p_bob_meas (float): The individual QBER probability for Bob's measurement.
        selected_state_id (int): The Bell state ID used for the simulation (1-4).

    Returns:
        tuple: (count_sifted_bit, count_error_bit)
            count_sifted_bit (int): The number of sifted bits obtained.
            count_error_bit (int): The number of mismatched bits (errors) in the sifted key.
    """
    if num_pairs_for_qiskit_circuit == 0:
        return 0, 0 # If no pairs to process, return 0 sifted bits and 0 errors

    # Generate the quantum circuit with the specified number of Bell pairs
    qc = generate_star_bell_circuit(num_pairs_for_qiskit_circuit, selected_state_id)
    
    # Apply the noise model based on Alice's and Bob's individual QBERs
    noise_model = apply_noise_model(
        p_alice_meas,
        p_bob_meas,
        num_pairs_for_qiskit_circuit
    )

    # Measure the qubits and get the bases used by Alice and Bob
    alice_bases, bob_bases, _ = measure_star_users(
        qc, num_pairs_for_qiskit_circuit, 0 # current_bob_id is not used in measure_star_users for logic
    )

    # Use Qiskit AerSimulator to run the circuit with noise
    backend = AerSimulator()
    result = backend.run(qc, shots=1, noise_model=noise_model).result()
    counts = result.get_counts()

    if not counts:
        # This can happen if all shots result in no counts (e.g., severe noise)
        return 0, 0 # No measurement results

    # Extract the measured bits string (Qiskit returns in reverse order)
    measured_bits_str = list(counts.keys())[0][::-1] 

    # Separate Alice's (even qubits) and Bob's (odd qubits) bits from the raw measurement string
    alice_bits_raw = ''.join([measured_bits_str[i] for i in range(0, len(measured_bits_str), 2)])
    bob_bits_raw = ''.join([measured_bits_str[i] for i in range(1, len(measured_bits_str), 2)])

    # Use the full raw bits and bases as they correspond to the simulated circuit size
    alice_bits = alice_bits_raw
    bob_bits = bob_bits_raw
    
    # Sift the key based on matching bases and Bell state corrections
    sifted_alice, sifted_bob = sift_key_multi(
        alice_bits, bob_bits, alice_bases, bob_bases, selected_state_id
    )

    # Calculate the number of sifted bits and the number of errors
    count_sifted_bit = len(sifted_bob)
    count_error_bit = 0
    for i in range(count_sifted_bit):
        if sifted_alice[i] != sifted_bob[i]:
            count_error_bit += 1

    return count_sifted_bit, count_error_bit