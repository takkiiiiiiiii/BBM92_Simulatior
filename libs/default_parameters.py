# # Aperture of radius (Receiver radis in meters) (m)
# a = 0.75

# # average number of photon
# n_s = 0.1

# # average number of photon of decoy-state signal
# # n_d = 0.09

# # len_wave : Optical wavelength (μm)
# wavelength = 0.85e-6

# # altitude ground station (m)
# h_OGS = 10

# # satellite altitude (m)
# h_s = 550 * 1000

# # atmosphere altitude (m)
# h_atm = 20 * 1000

# # Half-divergence angle (rad)
# theta_rad = 10e-6

# # rms wind_speed
# v_wind = 21

# # mu_x, mu_y: Mean values of pointing error in x and y directions (m)
# mu_x = 0
# mu_y = 0

# # Standard deviation of pointing error in x and y directions (radians)
# sigma_theta_x = theta_rad/8
# sigma_theta_y = theta_rad/8

# # the error rate of the background
# e_0 = 0.5

# # the background rate which includes the detector dark count
# # and other background contributions
# p_dark = 1e-5

# # After-pulsing probability
# # p_AP = 2/100

# # Probability of the polarisation errors
# e_pol = 1/100

# # Background count rates at Alice’s and Bob’s
# # Y0_A = 6.02e-6
# # Y0_B = 6.02e-6

# # Error rate of the detector (same as e_pol)
# e_d = 1/100

# # One half of the expected photon pair number n_s
# lambda_signal = 0.5 * n_s

# Yudai # Default parameters for the simulation
a = 0.75 # Aperture radius of the receiver telescope (m)
n_s = 0.1 # Mean photon number per pulse (for single photon source)
lambda_signal = 0.5 * n_s
wavelength = 0.85e-6 # Wavelength of light (m)
h_OGS = 10 # Height of Optical Ground Station (m)
h_s = 500 * 1000 # Height of Satellite (m)
h_atm = 20 * 1000 # Height of atmosphere (m)
theta_rad = 5e-6 # Beam divergence angle (radians)
v_wind = 21 # Wind speed (m/s) for Cn2 profile
mu_x = 0 # Mean pointing error in x (m)
mu_y = 0 # Mean pointing error in y (m)
sigma_theta_x = theta_rad/6 # Standard deviation of pointing error in x (radians)
sigma_theta_y = theta_rad/6 # Standard deviation of pointing error in y (radians)
e_0 = 0.5 # Detector efficiency
p_dark = 1e-5 # Dark count probability
e_pol = 1/100 # Polarization misalignment error


# Yuma # Default parameters for the simulation
# a = 0.75 # Aperture radius of the receiver telescope (m)
# # n_s = 0.053 # Mean photon number per pulse (for single photon source)
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