# Aperture of radius (Receiver radis in meters) (m)
a = 0.75

# average number of photon
n_s = 1.0

# average number of photon of decoy-state signal
n_d = 0.09

# len_wave : Optical wavelength (μm)
wavelength = 0.85e-6

# altitude ground station (m)
h_OGS = 10

# satellite altitude (m)
h_s = 550 * 1000

# atmosphere altitude (m)
h_atm = 20 * 1000

# Half-divergence angle (rad)
theta_rad = 10e-6

# rms wind_speed
v_wind = 21

# mu_x, mu_y: Mean values of pointing error in x and y directions (m)
mu_x = 0
mu_y = 0
sigma_theta_x = theta_rad/8
sigma_theta_y = theta_rad/8

# the error rate of the background
e_0 = 0.5

# the background rate which includes the detector dark count
# and other background contributions
p_dark = 1e-5

# After-pulsing probability
p_AP = 2/100

# Probability of the polarisation errors
e_pol = 1/100

# Background count rates at Alice’s and Bob’s
Y0_A = 6.02e-6
Y0_B = 6.02e-6

# Error rate of the detector
e_d = 1.5 / 100

# One half of the expected photon pair number n_s
lambda_signal = n_s / 2