import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

# Import all functions and global parameters from tool.py
from tool import (
    append_bell_state, generate_star_bell_circuit, measure_star_users,
    sift_key_multi, apply_noise_model, compute_atm_loss, rytov_variance,
    Cn2_profile, compute_slant_distance, equivalent_beam_width_squared,
    rvs_LN_fading, rvs_pointing_err, _get_precalculated_channel_params,
    generate_insta_eta_optimized, binary_entropy, P, bbm92simulation,
    # Global parameters
    a, n_s, wavelength, h_OGS, h_s, h_atm, theta_rad, v_wind, mu_x, mu_y,
    sigma_theta_x, sigma_theta_y, e_0, p_dark, e_pol
)

from qchannel_model import (
    transmitivity_pdf,
    compute_sigma_mod,
    sigma_to_variance,
)
from scipy.special import erf, erfc
from scipy.integrate import dblquad

def calculate_sigma_mod_wrapper(mu_x_val, mu_y_val, sigma_x_val, sigma_y_val):
    """
    Wrapper for compute_sigma_mod from libs.qchannel_model.
    Calculates sigma_mod.
    """
    return compute_sigma_mod(mu_x_val, mu_y_val, sigma_x_val, sigma_y_val)

def calculate_phi_mod_wrapper(w_Leq_val, sigma_mod_val):
    """
    Wrapper for sigma_to_variance from libs.qchannel_model.
    Calculates phi_mod.
    """
    return sigma_to_variance(sigma_mod_val, w_Leq_val)

def calculate_phi_x_y_wrapper(w_Leq_val, sigma_x_val, sigma_y_val):
    """
    Calculates phi_x and phi_y.
    """
    phi_x_val = w_Leq_val / (2 * sigma_x_val + 1e-15) if sigma_x_val != 0 else float('inf')
    phi_y_val = w_Leq_val / (2 * sigma_y_val + 1e-15) if sigma_y_val != 0 else float('inf')
    return phi_x_val, phi_y_val

def calculate_A_mod_wrapper(A0_val, phi_mod_val, phi_x_val, phi_y_val, sigma_x_val, sigma_y_val):
    """
    Calculates A_mod based on the given formula.
    This function specifically mirrors the formula from the image, using explicit terms,
    as mod_jitter in qchannel_model might have different internal structure.
    """
    mu_x_val = mu_x
    mu_y_val = mu_y

    epsilon = 1e-15

    # Denominators
    phi_mod_denom = (2 * phi_mod_val**2 + epsilon)
    phi_x_denom = (2 * phi_x_val**2 + epsilon)
    phi_y_denom = (2 * phi_y_val**2 + epsilon)
    sigma_x_phi_x_denom = (2 * sigma_x_val**2 * phi_x_val**2 + epsilon)
    sigma_y_phi_y_denom = (2 * sigma_y_val**2 * phi_y_val**2 + epsilon)

    exp_term_corrected = (1 / phi_mod_denom) \
                         - (1 / phi_x_denom) \
                         - (1 / phi_y_denom) \
                         - (mu_x_val**2 / sigma_x_phi_x_denom) \
                         - (mu_y_val**2 / sigma_y_phi_y_denom)
    
    exp_term_corrected = np.clip(exp_term_corrected, None, np.log(np.finfo(float).max) - 1)

    return A0_val * np.exp(exp_term_corrected)

def calculate_chi_wrapper(sigma_R_val, phi_mod_val):
    """
    Calculates chi.
    """
    return 0.5 * sigma_R_val**2 * (1 + 2 * phi_mod_val**2)

# --- Q_lambda (式9) の関数定義 ---
def Q_lambda_func(eta_A, eta_B, lambda_val, p_dark):
    term1_denominator = (1 + eta_A * lambda_val)**2
    term2_denominator = (1 + eta_B * lambda_val)**2
    term3_denominator = (1 + eta_A * lambda_val + eta_B * lambda_val - eta_A * eta_B * lambda_val)**2

    min_denom = 1e-15

    q_lambda = 1.0 \
             - (1.0 - p_dark) / (term1_denominator + min_denom) \
             - (1.0 - p_dark) / (term2_denominator + min_denom) \
             + ((1.0 - p_dark) * (1.0 - p_dark)) / (term3_denominator + min_denom)
    
    return q_lambda

# --- NEW: Q_lambda の修正版 ---
def Q_lambda_func_modified(eta_A, eta_B, lambda_val, p_dark, e_0_val, e_pol_val):
    """
    Q(eta_A, eta_B, lambda, p_dark) の修正版。
    """
    # 既存の Q_lambda_func の結果を取得
    q_lambda_original = Q_lambda_func(eta_A, eta_B, lambda_val, p_dark)

    # 追加項の分母
    denom_term1 = (1 + eta_A * lambda_val)
    denom_term2 = (1 + eta_B * lambda_val)
    denom_term3 = (1 + eta_A * lambda_val + eta_B * lambda_val - eta_A * eta_B * lambda_val)
    
    min_denom = 1e-15 # ゼロ除算防止

    # 分母がゼロに近づかないようにクリッピング
    denom_product = (denom_term1 + min_denom) * (denom_term2 + min_denom) * (denom_term3 + min_denom)
    
    # 追加項
    additional_term = (2 * (e_0_val - e_pol_val) * eta_A * eta_B * lambda_val * (1 + lambda_val)) / (denom_product)

    # 修正された Q の値
    q_modified = e_0_val * q_lambda_original - additional_term
    
    return q_modified

# --- NEW: 積分する関数 (Q_lambda * transmitivity_pdf_A * transmitivity_pdf_B) ---
def integrand_func(eta_B, eta_A, lambda_val_param, p_dark_param,
                   # Alice parameters for transmitivity_pdf
                   mu_x_A, mu_y_A, sigma_x_A, sigma_y_A, zenith_angle_rad_A,
                   w_L_A, w_Leq_A, tau_zen_A, varphi_mod_A, wavelength_A, h_OGS_A,
                   h_atm_A, Cn2_profile_A, a_A,
                   # Bob parameters for transmitivity_pdf
                   mu_x_B, mu_y_B, sigma_x_B, sigma_y_B, zenith_angle_rad_B,
                   w_L_B, w_Leq_B, tau_zen_B, varphi_mod_B, wavelength_B, h_OGS_B,
                   h_atm_B, Cn2_profile_B, a_B):
    
    Q_val = Q_lambda_func(eta_A, eta_B, lambda_val_param, p_dark_param)
    
    f_A = transmitivity_pdf(
        eta_A, mu_x_A, mu_y_A, sigma_x_A, sigma_y_A, zenith_angle_rad_A,
        w_L_A, w_Leq_A, tau_zen_A, varphi_mod_A, wavelength_A, h_OGS_A,
        h_atm_A, Cn2_profile_A, a_A
    )
    
    f_B = transmitivity_pdf(
        eta_B, mu_x_B, mu_y_B, sigma_x_B, sigma_y_B, zenith_angle_rad_B,
        w_L_B, w_Leq_B, tau_zen_B, varphi_mod_B, wavelength_B, h_OGS_B,
        h_atm_B, Cn2_profile_B, a_B
    )
    
    return Q_val * f_A * f_B

# --- NEW: 積分する関数 (修正版 Q_lambda * transmitivity_pdf_A * transmitivity_pdf_B) ---
def integrand_func_modified(eta_B, eta_A, lambda_val_param, p_dark_param, e_0_val_param, e_pol_val_param,
                   # Alice parameters for transmitivity_pdf
                   mu_x_A, mu_y_A, sigma_x_A, sigma_y_A, zenith_angle_rad_A,
                   w_L_A, w_Leq_A, tau_zen_A, varphi_mod_A, wavelength_A, h_OGS_A,
                   h_atm_A, Cn2_profile_A, a_A,
                   # Bob parameters for transmitivity_pdf
                   mu_x_B, mu_y_B, sigma_x_B, sigma_y_B, zenith_angle_rad_B,
                   w_L_B, w_Leq_B, tau_zen_B, varphi_mod_B, wavelength_B, h_OGS_B,
                   h_atm_B, Cn2_profile_B, a_B):
    
    Q_val_modified = Q_lambda_func_modified(eta_A, eta_B, lambda_val_param, p_dark_param, e_0_val_param, e_pol_val_param)
    
    f_A = transmitivity_pdf(
        eta_A, mu_x_A, mu_y_A, sigma_x_A, sigma_y_A, zenith_angle_rad_A,
        w_L_A, w_Leq_A, tau_zen_A, varphi_mod_A, wavelength_A, h_OGS_A,
        h_atm_A, Cn2_profile_A, a_A
    )
    
    f_B = transmitivity_pdf(
        eta_B, mu_x_B, mu_y_B, sigma_x_B, sigma_y_B, zenith_angle_rad_B,
        w_L_B, w_Leq_B, tau_zen_B, varphi_mod_B, wavelength_B, h_OGS_B,
        h_atm_B, Cn2_profile_B, a_B
    )
    
    return Q_val_modified * f_A * f_B


def calculate_skr(sifted_bit_percentage, qber):
    """
    Calculates the Secret Key Rate (SKR) in Kbit/s.
    sifted_bit_percentage: Sifted Bit (%) as a percentage (e.g., 5.0 for 5%)
    qber: QBER as a decimal (e.g., 0.01 for 1%)
    """
    R_key_raw = 10**9 * (sifted_bit_percentage) # Convert percentage to probability

    # H(QBER) の計算 (QBERは0-1の小数に変換)
    QBER_for_H = qber
    
    # QBERが0または1に非常に近い場合、log2(0)によるエラーを回避するためのクリッピング
    QBER_for_H = np.clip(QBER_for_H, 1e-10, 1 - 1e-10)

    H_QBER = binary_entropy(QBER_for_H)
    
    skr = R_key_raw * (1 - 2.1*H_QBER)

    skr = np.maximum(0, skr) # Clip SKR to be non-negative
    
    return skr / 1000 # Convert to Kbit/s

# --- Main simulation function ---
def main():
    # --- Simulation settings ---
    selected_state_id = 3
    num_timeslots = 10000
    array_size = 10**5
    
    # Average photon numbers to simulate
    ns_values = np.array([0.1, 0.2, 0.4, 0.8, 1.0])

    # Bob Configurations
    bob_configs = [
        {"id": 1, "zenith_angle_deg": 10},
        {"id": 2, "zenith_angle_deg": 45}
    ]
    
    # Store results for plotting
    results_bob1_sim_skr = []
    results_bob1_theo_skr = []
    results_bob2_sim_skr = []
    results_bob2_theo_skr = []
    results_alice_sim_skr = []
    results_alice_theo_skr = []

    # New lists to store QBER results
    results_bob1_sim_qber = []
    results_bob1_theo_qber = []
    results_bob2_sim_qber = []
    results_bob2_theo_qber = []


    print(f"--- QKD Simulation Started (Time-Slot Method) ---")
    print(f"Total number of time slots per zenith angle: {num_timeslots}")
    print(f"Photon number distribution array size (pulses per timeslot): {array_size}")
    print(f"Bell State ID used: {selected_state_id}\n")
    print("--- Alice Configuration ---")
    print(f"  Zenith Angle: 30 degrees, Tau_zen: 0.91")
    print("----------------------------------------")

    # --- Pre-calculate static channel parameters for Alice ---
    alice_precalc_params = _get_precalculated_channel_params(30, 0.91)

    # Loop through different n_s values
    for current_ns in ns_values:
        print(f"\n" + "#"*60)
        print(f"### Simulating for Average Photon Number (n_s): {current_ns:.2f} ###")
        print(f"#"*60)

        # To store SKR for each Bob for the current n_s
        skr_sim_for_current_ns = []
        skr_theo_for_current_ns = []

        for bob_config in bob_configs:
            bob_id = bob_config["id"]
            bob_zenith_deg = bob_config["zenith_angle_deg"]

            print(f"\n" + "="*60)
            print(f"=== Simulating for Bob {bob_id}'s Zenith Angle: {bob_zenith_deg} degrees ===")
            print(f"="*60)

            # Re-initialize accumulators for each new Bob zenith angle
            total_sifted_bit_overall = 0
            total_error_bit_overall = 0
            total_effective_pairs_processed_by_qiskit = 0
            total_pulses_generated_overall = 0 

            # Pre-calculate static channel parameters for the current Bob zenith angle
            bob_precalc_params = _get_precalculated_channel_params(bob_zenith_deg, 0.91) # tau_zenは固定

            # --- Time slot loop ---
            for ts in range(num_timeslots):
                # Dynamically calculate instantaneous channel parameters for Alice and current Bob
                insta_eta_alice = generate_insta_eta_optimized(alice_precalc_params)
                insta_eta_bob = generate_insta_eta_optimized(bob_precalc_params)

                # --- Photon number distribution and detection simulation ---
                lambda_val = 0.5 * current_ns # Use current_ns for lambda_val
                
                n_values = np.arange(11)
                p_values_raw = [P(n, lambda_val) for n in n_values[:10]]
                p_values = p_values_raw[:]
                p_values.append(1.0 - sum(p_values))
                p_values = np.array(p_values)
                p_values = np.clip(p_values, 0, 1)
                p_values = p_values / np.sum(p_values)

                photon_numbers_array = np.random.choice(n_values, size=array_size, p=p_values)
                count_array = Counter(photon_numbers_array)
                
                n_detect_event = [0] * len(n_values)
                
                current_timeslot_effective_pairs = 0
                total_pulses_generated_overall += array_size

                for i in range(len(n_detect_event)):
                    num_pulses_with_n_photons = count_array[i]
                    
                    p_detect_alice_for_n = (1 - (1 - p_dark) * ((1 - insta_eta_alice)**i))
                    p_detect_bob_for_n = (1 - (1 - p_dark) * ((1 - insta_eta_bob)**i))
                    
                    Yi = p_detect_alice_for_n * p_detect_bob_for_n
                    
                    for j in range(num_pulses_with_n_photons):
                        if random.random() < Yi:
                            n_detect_event[i] += 1
                    
                    current_timeslot_effective_pairs += n_detect_event[i]

                total_effective_pairs_processed_by_qiskit += current_timeslot_effective_pairs

                if current_timeslot_effective_pairs == 0:
                    continue

                # --- Calculate n-dependent QBER and call BBM92 simulation ---
                for i in range(len(n_detect_event)):
                    if n_detect_event[i] > 0:
                        p_detect_alice_for_n = (1 - (1 - p_dark) * ((1 - insta_eta_alice)**i))
                        p_detect_bob_for_n = (1 - (1 - p_dark) * ((1 - insta_eta_bob)**i))
                    
                        yi = p_detect_alice_for_n * p_detect_bob_for_n
                        
                        current_alice_qber = 0.0 # Alice's QBER often assumed 0 for ideal source
                        # This Bob QBER formula seems complex and n-dependent.
                        current_bob_qber = e_0 - ((2*(e_0-e_pol)/((i+1)*yi))*((1-(1-insta_eta_alice)**(i+1)*(1-insta_eta_bob)**(i+1))/(1-(1-insta_eta_alice)*(1-insta_eta_bob))-((1-insta_eta_alice)**(i+1)-(1-insta_eta_bob)**(i+1))/(insta_eta_bob-insta_eta_alice)))
                        
                        # Handle potential division by zero or NaN if insta_eta_bob == insta_eta_alice
                        if abs(insta_eta_bob - insta_eta_alice) < 1e-15:
                            if insta_eta_bob == 0: # Both are 0
                                term2 = 0 # (1-(1-0)^(i+1))/(0) -> 0
                            else: # Both are non-zero and equal
                                term2 = (i+1) * (1-insta_eta_alice)**i # L'Hopital's rule limit of (x^k-y^k)/(x-y) as x->y is k*x^(k-1)
                                
                            current_bob_qber = e_0 - ((2*(e_0-e_pol)/((i+1)*yi)) * (
                                (1-(1-insta_eta_alice)**(i+1)*(1-insta_eta_bob)**(i+1))/(1-(1-insta_eta_alice)*(1-insta_eta_bob)) - term2
                            ))
                            
                        count_sifted, count_error = bbm92simulation(
                            n_detect_event[i],
                            current_alice_qber,
                            current_bob_qber,
                            selected_state_id
                        )
                        total_sifted_bit_overall += count_sifted
                        total_error_bit_overall += count_error
            print(f'total_sifted_bit_length: {total_sifted_bit_overall}, total_err_num: {total_error_bit_overall}')

            # --- Post-simulation QBER verification (Overall for current zenith angle) ---
            print(f"\n--- Results for Bob {bob_id}'s Zenith Angle: {bob_zenith_deg} degrees ---")
            
            average_simulated_sifted_bits = total_sifted_bit_overall / (array_size * num_timeslots)
            average_simulated_error_bits = total_error_bit_overall / (array_size * num_timeslots)

            print(f"  Sifted Bits (%) : {average_simulated_sifted_bits*100:.8f}")
            print(f"  Error  Bits (%) : {average_simulated_error_bits*100:.8f}")

            # --- Theoretical Calculations ---
            zenith_angle_rad_alice = np.deg2rad(30)
            sigma_R_alice = np.sqrt(alice_precalc_params['sigma_R_squared'])
            w_Leq_alice = np.sqrt(alice_precalc_params['w_Leq_squared'])
            w_L_alice = alice_precalc_params['w_L_beam']
            tau_zen_alice = 0.91
            slant_distance_alice = alice_precalc_params['slant_distance']
            sigma_x_alice = sigma_theta_x * slant_distance_alice
            sigma_y_alice = sigma_theta_y * slant_distance_alice

            zenith_angle_rad_bob = np.deg2rad(bob_zenith_deg)
            sigma_R_bob = np.sqrt(bob_precalc_params['sigma_R_squared'])
            w_Leq_bob = np.sqrt(bob_precalc_params['w_Leq_squared'])
            w_L_bob = bob_precalc_params['w_L_beam']
            tau_zen_bob = 0.91
            slant_distance_bob = bob_precalc_params['slant_distance']
            sigma_x_bob = sigma_theta_x * slant_distance_bob
            sigma_y_bob = sigma_theta_y * slant_distance_bob

            sigma_mod_alice_val = compute_sigma_mod(mu_x, mu_y, sigma_x_alice, sigma_y_alice)
            varphi_mod_alice_val = sigma_to_variance(sigma_mod_alice_val, w_Leq_alice)
            
            sigma_mod_bob_val = compute_sigma_mod(mu_x, mu_y, sigma_x_bob, sigma_y_bob)
            varphi_mod_bob_val = sigma_to_variance(sigma_mod_bob_val, w_Leq_bob)

            lower_bound = 0
            upper_bound = 1

            # Theoretical Gain
            integration_args_gain = (
                lambda_val, p_dark, # Use lambda_val (current_ns) here

                mu_x, mu_y, sigma_x_alice, sigma_y_alice, zenith_angle_rad_alice,
                w_L_alice, w_Leq_alice, tau_zen_alice, varphi_mod_alice_val, wavelength, h_OGS,
                h_atm, Cn2_profile, a,

                mu_x, mu_y, sigma_x_bob, sigma_y_bob, zenith_angle_rad_bob,
                w_L_bob, w_Leq_bob, tau_zen_bob, varphi_mod_bob_val, wavelength, h_OGS,
                h_atm, Cn2_profile, a
            )

            avg_overall_gain, integral_abserr_gain = dblquad(
                integrand_func,
                lower_bound, upper_bound,
                lambda eta_A: lower_bound,
                lambda eta_A: upper_bound,
                args=integration_args_gain
            )
            avg_overall_gain_scaled = avg_overall_gain / 2 # Convert to percentage

            # Theoretical Error
            integration_args_error = (
                lambda_val, p_dark, e_0, e_pol, # Use lambda_val (current_ns) and global e_0, e_pol

                mu_x, mu_y, sigma_x_alice, sigma_y_alice, zenith_angle_rad_alice,
                w_L_alice, w_Leq_alice, tau_zen_alice, varphi_mod_alice_val, wavelength, h_OGS,
                h_atm, Cn2_profile, a,

                mu_x, mu_y, sigma_x_bob, sigma_y_bob, zenith_angle_rad_bob,
                w_L_bob, w_Leq_bob, tau_zen_bob, varphi_mod_bob_val, wavelength, h_OGS,
                h_atm, Cn2_profile, a
            )

            avg_overall_error_modified, integral_abserr_error = dblquad(
                integrand_func_modified,
                lower_bound, upper_bound,
                lambda eta_A: lower_bound,
                lambda eta_A: upper_bound,
                args=integration_args_error
            )
     

            # --- QBER Calculations ---
            calculated_qber_overall = 0.0
            if total_sifted_bit_overall > 0:
                calculated_qber_overall = total_error_bit_overall / total_sifted_bit_overall
            else:
                calculated_qber_overall = 0.5 # Default to 0.5 if no bits, implies random

            theoretical_qber = 0.0
            if avg_overall_gain > 1e-15: # Use raw gain, not scaled, for denominator
                theoretical_qber = avg_overall_error_modified / avg_overall_gain
            else:
                theoretical_qber = 0.5 # Default to 0.5 if theoretical gain is near zero

            print(f"  Overall QBER (Simulated): {calculated_qber_overall:.8f}")
            print(f"  Overall Gain (Simulated): {average_simulated_sifted_bits*100:.8f}")
            print(f"  Overall QBER (Theoretical): {theoretical_qber:.8f}")
            print(f"  Overall Gain (Theoretical): {avg_overall_gain_scaled*100:.8f}")
            print("----------------------------------------")

            # Store QBER results
            if bob_id == 1:
                results_bob1_sim_qber.append(calculated_qber_overall)
                results_bob1_theo_qber.append(theoretical_qber)
            elif bob_id == 2:
                results_bob2_sim_qber.append(calculated_qber_overall)
                results_bob2_theo_qber.append(theoretical_qber)

            # --- Calculate SKR ---
            skr_sim = calculate_skr(average_simulated_sifted_bits, calculated_qber_overall)/2

            skr_theo = calculate_skr(avg_overall_gain_scaled, theoretical_qber)/2

            print(f"  SKR (Simulated)  : {skr_sim:.4f} Kbit/s")
            print(f"  SKR (Theoretical): {skr_theo:.4f} Kbit/s")

            # Store SKR results for the current Bob
            if bob_id == 1:
                results_bob1_sim_skr.append(skr_sim)
                results_bob1_theo_skr.append(skr_theo)
            elif bob_id == 2:
                results_bob2_sim_skr.append(skr_sim)
                results_bob2_theo_skr.append(skr_theo)
            
            skr_sim_for_current_ns.append(skr_sim)
            skr_theo_for_current_ns.append(skr_theo)

        # Calculate Alice's SKR for the current n_s (average of all Bobs)
        if len(skr_sim_for_current_ns) > 0:
            avg_skr_sim_alice = sum(skr_sim_for_current_ns) 
            avg_skr_theo_alice = sum(skr_theo_for_current_ns)
            results_alice_sim_skr.append(avg_skr_sim_alice)
            results_alice_theo_skr.append(avg_skr_theo_alice)
            print(f"\n--- Alice's SKR for n_s = {current_ns:.2f} ---")
            print(f"  SKR (Simulated)  : {avg_skr_sim_alice:.4f} Kbit/s")
            print(f"  SKR (Theoretical): {avg_skr_theo_alice:.4f} Kbit/s")
            print("------------------------------------------")


    # --- Plotting the SKR results ---
    plt.figure(figsize=(12, 8))

    plt.plot(ns_values, results_bob1_sim_skr, 'o', color='blue', label=f'Bob 1 (Zenith 10°) Simulated SKR')
    plt.plot(ns_values, results_bob1_theo_skr, '--', color='blue', label=f'Bob 1 (Zenith 10°) Theoretical SKR')

    plt.plot(ns_values, results_bob2_sim_skr, 'o', color='red', label=f'Bob 2 (Zenith 45°) Simulated SKR')
    plt.plot(ns_values, results_bob2_theo_skr, '--', color='red', label=f'Bob 2 (Zenith 45°) Theoretical SKR')
    
    plt.plot(ns_values, results_alice_sim_skr, 'o', color='green', label=f'Alice Simulated SKR')
    plt.plot(ns_values, results_alice_theo_skr, '--', color='green', label=f'Alice Theoretical SKR')

    plt.xlabel('Average Photon Number ($n_s$)')
    plt.ylabel('Secret Key Rate (Kbit/s)')
    plt.title('Secret Key Rate vs. Average Photon Number for Multiple Bobs and Alice')
    plt.legend()
    plt.grid(True)
    plt.xscale('linear') # Ensure linear scale for n_s
    plt.xticks(ns_values) # Set x-ticks to the specific values
    plt.tight_layout()
    plt.show()

    # --- Plotting the QBER results ---
    plt.figure(figsize=(12, 8))

    plt.plot(ns_values, results_bob1_sim_qber, 'o', color='cyan', label=f'Bob 1 (Zenith 10°) Simulated QBER')
    plt.plot(ns_values, results_bob1_theo_qber, '--', color='cyan', label=f'Bob 1 (Zenith 10°) Theoretical QBER')

    plt.plot(ns_values, results_bob2_sim_qber, 'o', color='magenta', label=f'Bob 2 (Zenith 45°) Simulated QBER')
    plt.plot(ns_values, results_bob2_theo_qber, '--', color='magenta', label=f'Bob 2 (Zenith 45°) Theoretical QBER')

    plt.xlabel('Average Photon Number ($n_s$)')
    plt.ylabel('Quantum Bit Error Rate (QBER)')
    plt.title('QBER vs. Average Photon Number for Different Bob Zenith Angles')
    plt.legend()
    plt.grid(True)
    plt.xscale('linear') # Ensure linear scale for n_s
    plt.xticks(ns_values) # Set x-ticks to the specific values
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()