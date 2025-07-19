import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erfc, erf
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from simulation_tools import rvs_pointing_err, rvs_LN_fading
# from scipy.stats import lognorm


def rvs_pointing_err(
        mu_x, mu_y, sigma_theta_x, sigma_theta_y,
        slant_distance, theta_rad, a, w_Leq_squared, size=1):
    sigma_x = sigma_theta_x * slant_distance
    sigma_y = sigma_theta_y * slant_distance

    x = np.random.normal(loc=mu_x, scale=sigma_x, size=size)
    y = np.random.normal(loc=mu_y, scale=sigma_y, size=size)

    r = np.sqrt(x**2 + y**2)

    w_L = slant_distance * theta_rad
    nu = (np.sqrt(np.pi) * a) / (np.sqrt(2) * w_L)
    A0 = erf(nu)**2
    eta_p = A0 * np.exp(-(2*r**2)/(w_Leq_squared))

    return eta_p



def rvs_LN_fading(sigma_R_squared, size=1):
    shape_param = np.sqrt(sigma_R_squared)
    # I_a_random = 0
    # while (I_a_random <= 0):
    #     I_a_random = lognorm.rvs(
    #         shape_param, loc=-sigma_R_squared/2, size=size
    #     )
    I_a_random = np.random.lognormal(
        mean=-sigma_R_squared/2, sigma=shape_param, size=size)

    return I_a_random

def transmitivity_pdf(
        eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
        w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
        h_atm, Cn2_profile, a):

    eta_l = compute_atm_loss(tau_zen, zenith_angle_rad)

    sigma_R_squared = rytov_variance(
        wavelength, zenith_angle_rad, h_OGS, h_atm, Cn2_profile)

    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_L, w_Leq, a)

    mu = (sigma_R_squared/2) * (1 + 2 * varphi_mod**2)

    term1 = (varphi_mod**2) / (2 * (A_mod * eta_l)**(varphi_mod**2))
    term2 = eta**(varphi_mod**2-1)
    term3 = erfc(
        (np.log(eta / (A_mod * eta_l)) + mu)
        / (np.sqrt(2) * np.sqrt(sigma_R_squared))
        )
    term4 = np.exp(
        (sigma_R_squared / 2) * varphi_mod**2 * (1 + varphi_mod**2)
    )
    # print(term1, term2, term3, term4)
    eta = term1 * term2 * term3 * term4

    return eta


def compute_atm_loss(tau_zen, zenith_angle_rad):
    tau_atm = tau_zen ** (1 / np.cos(zenith_angle_rad))
    if tau_atm > 1:
        raise ValueError("Atmospheric loss is larger than 1")

    return tau_atm


def rytov_variance(len_wave, zenith_angle_rad, h_OGS, h_atm, Cn2_profile):
    k = 2 * np.pi / len_wave
    sec_zenith = 1 / np.cos(zenith_angle_rad)

    def integrand(h):
        return Cn2_profile(h) * (h - h_OGS)**(5/6)

    integral, _ = quad(integrand, h_OGS, h_atm)

    sigma_R_squared = 2.25 * (k)**(7/6) * sec_zenith**(11/6) * integral

    return sigma_R_squared


def mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_L, w_Leq, a):
    A_0 = compute_A0(a, w_L)

    varphi_x = sigma_to_variance(sigma_x, w_Leq)
    varphi_y = sigma_to_variance(sigma_y, w_Leq)
    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)

    term1 = 1 / (varphi_mod**2)
    term2 = 1 / (2 * varphi_x**2)
    term3 = 1 / (2 * varphi_y**2)
    term4 = mu_x**2 / (2 * sigma_x**2 * varphi_x**2)
    term5 = mu_y**2 / (2 * sigma_y**2 * varphi_y**2)
    G = np.exp(term1 - term2 - term3 - term4 - term5)

    A_mod = A_0 * G

    return A_mod


def compute_A0(a, w_L):
    """_summary_

    Args:
        a (_type_): asparture radius
        w_L (_type_): beam waist at distance L
    """
    nu = (np.sqrt(np.pi) * a) / (np.sqrt(2) * w_L)
    A_0 = erf(nu)**2

    return A_0


def sigma_to_variance(sigma, w_Leq):
    return w_Leq/(2*sigma)


def compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y):
    numerator = (
        3 * mu_x**2 * sigma_x**4 +
        3 * mu_y**2 * sigma_y**4 +
        sigma_x**6 +
        sigma_y**6
    )
    sigma_mod_value = (numerator / 2) ** (1/3)
    return sigma_mod_value


def Cn2_profile(h, v_wind=21, Cn2_0=1e-13):
    term1 = 0.00594 * (v_wind/27)**2 * (1e-5 * h)**10 * np.exp(-h/1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)

    return term1 + term2 + term3


def qber_loss(e_0, p_dark, e_pol, p_AP, eta, n_s):
    denominator = (
        e_0 * (p_dark*(1+p_AP)) + (e_pol+e_0*p_AP) * (1-np.exp(-n_s*eta))
    )
    numerator = (p_dark*(1+p_AP)) + (1-np.exp(-n_s*eta)) * (1+p_AP)
    qber = denominator/numerator

    return qber


def weather_condition(tau_zen):
    if tau_zen == 0.91:
        return 'Clear sky', 23000  # H_atm for clear sky
    elif tau_zen == 0.85:
        return 'Slightly hazy', 15000  # H_atm for slightly hazy
    elif tau_zen == 0.75:
        return 'Noticeably hazy', 10000  # H_atm for noticeably hazy
    elif tau_zen == 0.53:
        return 'Poor visibility', 5000  # H_atm for poor visibility
    else:
        return 'Unknown condition', 10000  # Default value


# def compute_slant_distance(h_s, H_g, zenith_angle_rad):
#     return (h_s - H_g)/np.cos(zenith_angle_rad)

def compute_slant_distance(h_s, H_g, zenith_angle_rad):
    """Computes the slant distance between satellite and ground station."""
    delta_h = h_s - H_g
    if np.cos(zenith_angle_rad) == 0:
        return float('inf')
    horizontal_distance = delta_h * np.tan(zenith_angle_rad)
    return np.sqrt(delta_h**2 + horizontal_distance**2)


# def equivalent_beam_width_squared(a, w_L):
#     # w_L: beam radius at receiver before aperture clipping
#     nu = (np.sqrt(np.pi) * a) / (np.sqrt(2) * w_L)
#     numerator = np.sqrt(np.pi) * erf(nu)
#     denominator = 2 * nu * np.exp(-nu**2)
#     return w_L**2 * (numerator / denominator)


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

def compute_avg_qber(
        sigma_theta_x, sigma_theta_y, slant_distance,
        mu_x, mu_y, zenith_angle_rad, h_OGS, h_atm, w_L, tau_zen,
        Cn2_profile, a, e_0, p_dark, e_pol, p_AP, n_s, wavelength):
    sigma_x = sigma_theta_x * slant_distance
    sigma_y = sigma_theta_y * slant_distance

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    w_Leq = np.sqrt(w_Leq_squared)
    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)

    # def integrand(eta):
    #     term_1 = transmitivity_pdf(
    #         eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
    #         w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
    #         h_atm, Cn2_profile, a)
    #     term_2 = qber_loss(e_0, p_dark, e_pol, p_AP, eta, n_s)

    #     return term_1 * term_2

    def integrand_2(eta):
        term_1 = transmitivity_pdf(
            eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
            w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
            h_atm, Cn2_profile, a)
        term_2 = compute_yield(eta, n_s, p_dark, p_AP)

        return term_1 * term_2

    def integrand_3(eta):
        term_1 = transmitivity_pdf(
            eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
            w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
            h_atm, Cn2_profile, a)
        term_2 = (
            e_0 * (p_dark*(1+p_AP))
            + (e_pol+e_0*p_AP) * (1-np.exp(-n_s*eta))
        )

        return term_1 * term_2

    # res, _ = quad(integrand, 0, np.inf)

    avg_yield, _ = quad(integrand_2, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)

    avg_err_bits, _ = quad(integrand_3, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)

    return avg_err_bits/avg_yield, avg_yield

# 使用しない
def compute_Q_1_e_1_ex(
        sigma_theta_x, sigma_theta_y, slant_distance,
        mu_x, mu_y, zenith_angle_rad, h_OGS, h_atm, w_L, tau_zen,
        Cn2_profile, a, e_0, p_dark, e_pol, p_AP, n_s, n_d, wavelength):
    sigma_x = sigma_theta_x * slant_distance
    sigma_y = sigma_theta_y * slant_distance

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    w_Leq = np.sqrt(w_Leq_squared)
    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)

    def integrand_Q_1_LB(eta):
        term_1 = transmitivity_pdf(
            eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
            w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
            h_atm, Cn2_profile, a)
        term_2 = (
            (np.exp(-n_s) * n_s**2)
            / (n_s * n_d - n_d**2)
        )
        Q_mu = compute_yield(eta, n_s, p_dark, p_AP)
        Q_nu = compute_yield(eta, n_d, p_dark, p_AP)
        
        term_3 = (
            Q_nu * np.exp(n_d)
            - Q_mu * np.exp(n_s) * n_d**2/n_s**2
            - (p_dark*(1+p_AP)) * (n_s**2 - n_d**2)/n_s**2 
        )

        return term_1 * term_2 * term_3

    def integrand_e_1_UB(eta):
        term_1 = transmitivity_pdf(
            eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
            w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
            h_atm, Cn2_profile, a)

        # Q_nu = compute_yield(eta, n_d, p_dark, p_AP)
        term_2 = (
            (e_0 * (p_dark*(1+p_AP))
            + (e_pol+e_0*p_AP) * (1-np.exp(-n_d*eta))) * np.exp(n_d)
            - e_0 * (p_dark*(1+p_AP))
        )

        term_3 = (
            n_s * np.exp(-n_s)/n_d
        )

        return term_1 * term_2 * term_3

    Q_1_lb, _ = quad(integrand_Q_1_LB, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)

    e_1_temp, _ = quad(integrand_e_1_UB, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)

    e_1_ub = e_1_temp / Q_1_lb

    return Q_1_lb, e_1_ub


# 使用しない
def compute_SKR(
        qber, avg_yield, Q_1, e_1, sifting_coefficient=0.5, p_estimation=0.75,
        kr_efficiency=1, decoy_coefficient=0.5, rep_rate=1e9):
    term_1 = rep_rate * sifting_coefficient * p_estimation * decoy_coefficient
    term_2 = -avg_yield * kr_efficiency * entropy_func(qber)

    term_3 = Q_1 * (1 - entropy_func(e_1))

    if (term_2 + term_3) < 0:
        return 0
    else:
        return term_1 * (term_2 + term_3)

def compute_yield(eta, n_s, p_dark, p_AP):
    param_q = (p_dark*(1+p_AP)) + (1-np.exp(-n_s*eta))*(1+p_AP)
    return param_q


def entropy_func(p):
    if p == 0 or p == 1:
        return 0
    p = np.asarray(p)
    res = -p * np.log2(p) - (1-p) * np.log2(1-p)
    return res


def compute_SKR_BBM92(qber, avg_yield, rep_rate=1e9, sifting_coefficient=0.5, kr_efficiency=1.22):
    """
    Compute Secret Key Rate for BBM92 protocol.

    Parameters:
    - qber: float, Quantum Bit Error Rate (0 to 1)
    - avg_yield: float, average coincidence probability (i.e., Q_lambda)
    - rep_rate: float, source repetition rate [Hz]
    - sifting_coefficient: float, e.g., 0.5 for BBM92
    - kr_efficiency: float, error correction inefficiency factor (f)

    Returns:
    - skr: float, secret key rate [bit/s]
    """
    H2 = entropy_func(qber)
    R_key_bit = rep_rate * avg_yield * sifting_coefficient
    skr = R_key_bit * (1 - kr_efficiency * H2)
    return max(skr, 0)  # SKRが負になるのを防ぐ

def photon_number_probability(n, lambda_):
    """
    Calculates the probability P(n) of having n photons in a pulse
    """
    if n < 0:
        return 0.0
    numerator = (n + 1) * (lambda_ ** n)
    denominator = (1 + lambda_) ** (n + 2)
    return numerator / denominator

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


# def qber_ma_model(e0, ed, etaA, etaB, lambda_, Y0_A, Y0_B):
#     numerator = 2 * (e0 - ed) * etaA * etaB * lambda_ * (1 + lambda_)
#     denominator = (1 + etaA * lambda_) * (1 + etaB * lambda_) * (1 + etaA * lambda_ + etaB * lambda_ - etaA * etaB * lambda_)
#     Q_lambda = compute_Q_lambda(lambda_, etaA, etaB, Y0_A, Y0_B)
#     E_lambda = (e0 * Q_lambda - numerator / denominator) / Q_lambda
#     E_lambda = float(E_lambda)
#     return E_lambda


def compute_Q_lambda(lambda_, etaA, etaB, p_dark):
    """
    Calculates the overall gain Q_lambda from Ma et al. (Eq. 9, full version).
    Parameters:
        lambda_ (float): PDC parameter (lambda = sinh^2(χ), mu = 2*lambda)
        etaA (float): Total detection efficiency for Alice
        etaB (float): Total detection efficiency for Bob
        p_dark (float): Probability of dark counts or background noise
    Returns:
        Q_lambda (float): Overall gain (coincidence detection probability per pulse)
    """
    term1 = (1 - p_dark) / (1 + etaA * lambda_)**2
    term2 = (1 - p_dark) / (1 + etaB * lambda_)**2
    denominator_term3 = (1 + etaA * lambda_ + etaB * lambda_ - etaA * etaB * lambda_)**2
    term3 = ((1 - p_dark) * (1 - p_dark)) / denominator_term3

    Q_lambda = 1 - term1 - term2 + term3
    return Q_lambda


def yield_from_photon_number(n, p_dark, eta_A, eta_B):
    term_A = 1 - (1 - p_dark) * (1 - eta_A)**n
    term_B = 1 - (1 - p_dark) * (1 - eta_B)**n
    Yn = term_A * term_B
    return Yn

def pauli_x_application_probability_bob(n, e_0, e_pol, Yn, eta_A, eta_B):
    """
    Calculate Pauli X error rate (e_n) based on Ma et al. Eq (11)-like formula.

    Parameters:
    - n : int
        Photon pair index (number of photon pairs - 1)
    - e_0 : float
        Baseline error rate (e.g., due to misalignment, etc.)
    - Yn : float
        Yield for n+1 photon pairs (i.e., conditional detection probability)
    - eta_A : float
        Instantaneous transmittance for Alice
    - eta_B : float
        Instantaneous transmittance for Bob
    - e_d : float, optional
        Intrinsic detector error rate (default 0)

    Returns:
    - current_bob_qber : float
        Pauli X error probability for n+1 photon pairs
    """

    n_plus_1 = n + 1
    term1_numerator = 1 - (1 - eta_A)**n_plus_1 * (1 - eta_B)**n_plus_1
    term1_denominator = 1 - (1 - eta_A) * (1 - eta_B)
    term1 = term1_numerator / term1_denominator if term1_denominator != 0 else 0.0

    term2_numerator = (1 - eta_A)**n_plus_1 - (1 - eta_B)**n_plus_1
    term2_denominator = eta_B - eta_A
    term2 = term2_numerator / term2_denominator if term2_denominator != 0 else 0.0

    correction = (2 * (e_0 - e_pol) / (n_plus_1 * Yn)) * (term1 - term2)

    current_bob_qber = e_0 - correction
    return current_bob_qber


def compute_EQ_lambda(e_0, eta_A, eta_B, lambda_, p_dark, e_pol):
    """
    Q(eta_A, eta_B, lambda, p_dark) の修正版。
    """
    # 既存の Q_lambda_func の結果を取得
    Q_lambda = compute_Q_lambda(lambda_, eta_A, eta_B, p_dark)

    # 追加項の分母
    denom_term1 = (1 + eta_A * lambda_)
    denom_term2 = (1 + eta_B * lambda_)
    denom_term3 = (1 + eta_A * lambda_ + eta_B * lambda_ - eta_A * eta_B * lambda_)
    
    min_denom = 1e-15 # ゼロ除算防止

    # 分母がゼロに近づかないようにクリッピング
    denom_product = (denom_term1 + min_denom) * (denom_term2 + min_denom) * (denom_term3 + min_denom)
    
    # 追加項
    additional_term = (2 * (e_0 - e_pol) * eta_A * eta_B * lambda_ * (1 + lambda_)) / (denom_product)

    # 修正された Q の値
    EQ_lambda = e_0 * Q_lambda - additional_term

    return EQ_lambda


def compute_avg_qber_bbm92(
        sigma_theta_x, sigma_theta_y, slant_distance_A, slant_distance_B,
        mu_x, mu_y, zenith_angle_rad_A, zenith_angle_rad_B,
        h_OGS, h_atm, w_L_A, w_L_B, tau_zen_A, tau_zen_B,
        Cn2_profile, a, e_0, e_pol, p_dark, lambda_, wavelength):

    def channel_params(slant_distance, zenith_angle_rad, w_L, tau_zen):
        sigma_x = sigma_theta_x * slant_distance
        sigma_y = sigma_theta_y * slant_distance
        w_Leq_squared = equivalent_beam_width_squared(a, w_L)
        w_Leq = np.sqrt(w_Leq_squared)
        sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
        varphi_mod = sigma_to_variance(sigma_mod, w_Leq)
        return sigma_x, sigma_y, w_Leq, varphi_mod, w_L, tau_zen, zenith_angle_rad

    sigma_x_A, sigma_y_A, w_Leq_A, varphi_mod_A, w_L_A, tau_zen_A, zenith_angle_rad_A = channel_params(
        slant_distance_A, zenith_angle_rad_A, w_L_A, tau_zen_A
    )
    sigma_x_B, sigma_y_B, w_Leq_B, varphi_mod_B, w_L_B, tau_zen_B, zenith_angle_rad_B = channel_params(
        slant_distance_B, zenith_angle_rad_B, w_L_B, tau_zen_B
    )

    def integrand_gain(eta_A, eta_B):
        # Q_lambda = compute_Q_lambda(lambda_, eta_A, eta_B, p_dark)
        Q_lambda = Q_lambda_func(eta_A, eta_B, lambda_, p_dark)

        p_eta_A = transmitivity_pdf(
            eta_A, mu_x, mu_y, sigma_x_A, sigma_y_A, zenith_angle_rad_A,
            w_L_A, w_Leq_A, tau_zen_A, varphi_mod_A, wavelength,
            h_OGS, h_atm, Cn2_profile, a
        )
        p_eta_B = transmitivity_pdf(
            eta_B, mu_x, mu_y, sigma_x_B, sigma_y_B, zenith_angle_rad_B,
            w_L_B, w_Leq_B, tau_zen_B, varphi_mod_B, wavelength,
            h_OGS, h_atm, Cn2_profile, a
        )
        return Q_lambda * p_eta_A * p_eta_B

    def integrand_error(eta_A, eta_B):
        # EQ_lambda = compute_EQ_lambda(
        #     e_0, eta_A, eta_B, lambda_, p_dark, e_pol
        # )
        EQ_lambda = Q_lambda_func_modified(eta_A, eta_B, lambda_, p_dark, e_0, e_pol)
        # Q_lambda = compute_Q_lambda(lambda_, eta_A, eta_B, Y0_A, Y0_B)
        # E_lambda = qber_ma_model(e_0, e_d, eta_A, eta_B, lambda_, Y0_A, Y0_B)

        p_eta_A = transmitivity_pdf(
            eta_A, mu_x, mu_y, sigma_x_A, sigma_y_A, zenith_angle_rad_A,
            w_L_A, w_Leq_A, tau_zen_A, varphi_mod_A, wavelength,
            h_OGS, h_atm, Cn2_profile, a
        )
        p_eta_B = transmitivity_pdf(
            eta_B, mu_x, mu_y, sigma_x_B, sigma_y_B, zenith_angle_rad_B,
            w_L_B, w_Leq_B, tau_zen_B, varphi_mod_B, wavelength,
            h_OGS, h_atm, Cn2_profile, a
        )
        return EQ_lambda * p_eta_A * p_eta_B

    avg_yield, _ = dblquad(
        integrand_gain, 0, 1,
        lambda _: 0, lambda _: 1,
        epsabs=1e-9, epsrel=1e-9
    )
    avg_error, _ = dblquad(
        integrand_error, 0, 1,
        lambda _: 0, lambda _: 1,
        epsabs=1e-9, epsrel=1e-9
    )

    avg_qber = avg_error / avg_yield if avg_yield > 0 else 0
    return avg_qber, avg_yield, avg_error

    
def compute_insta_eta(tau_zen, zenith_angle_rad, slant_path, mu_x, mu_y, sigma_theta_x, sigma_theta_y, a, w_Leq_squared_alice, theta_rad, sigma_R_squared):
    eta_ell = compute_atm_loss(tau_zen, zenith_angle_rad)
    eta_p = rvs_pointing_err(
        mu_x, mu_y, sigma_theta_x, sigma_theta_y,
        slant_path, theta_rad, a, w_Leq_squared_alice, size=1
    )[0] 
    I_a_alice = rvs_LN_fading(sigma_R_squared, size=1)[0]  
    insta_eta = eta_ell * I_a_alice * eta_p
    return insta_eta


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
