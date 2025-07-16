import numpy as np


class LEOsatellite:

    def __init__(self, filename):
        with open(filename) as f:
            contents = f.readlines()

        # Day
        self.epoch_time = float(contents[0][18:32]) - float(contents[0][18:23])
        # self.epoch_time = self.epoch_time * 24 * 3600
        self.inclination = float(contents[1][8:16])
        self.ascension = float(contents[1][17:25])
        self.eccentricity = float(contents[1][26:33])/10**7
        self.perigee = float(contents[1][34:42])
        self.mean_anomaly = float(contents[1][43:51])
        self.mean_motion = float(contents[1][52:63])

    def computeGeometricWithUser(
            self, year, day, hour, minute, second, UTC,
            longitude, latitude, elevation
    ):
        current_time_second = (hour - UTC)*3600 + minute*60 + second

        time_after_epoch = current_time_second - self.epoch_time * 24 * 3600

        current_time_day = self.epoch_time + (time_after_epoch/3600)/24

        inclination = self.inclination * np.pi/180

        perigee = self.perigee * np.pi/180

        ascension = self.ascension * np.pi/180

        n_s = (self.mean_motion * 2 * np.pi)/86400

        mean_anomaly = self.mean_anomaly * np.pi/180 + n_s*time_after_epoch
        mean_anomaly = np.mod(mean_anomaly, 360)

        longitude = longitude * np.pi/180

        latitude = latitude * np.pi/180

        # Equatorial raidus of the Earth (km)
        R_e = 6378.137

        eccentric_anomaly = (
            mean_anomaly + self.eccentricity*np.sin(mean_anomaly)
            + (self.eccentricity**2/2)*np.sin(2*mean_anomaly)
            + (self.eccentricity**3/8) * (
                3 * np.sin(3*mean_anomaly) - np.sin(mean_anomaly))
        )

        # the graviational constant in Km^3/s^2
        muy = 398600.5

        semi_major_axis = (muy/(n_s**2))**(1/3)

        # r = semi_major_axis * (
        #     1 - eccentricity * np.cos(eccentric_anomaly))

        x_0 = semi_major_axis * (np.cos(eccentric_anomaly) - self.eccentricity)

        y_0 = (
            semi_major_axis
            * np.sqrt(1 - self.eccentricity**2)
            * np.sin(eccentric_anomaly)
        )

        z_0 = 0

        orbital = np.array([x_0, y_0, z_0])

        # Transformation to inertial coordinates
        matrix_trans = np.array([
            [
                np.cos(perigee)*np.cos(ascension)
                - np.sin(perigee)*np.cos(inclination)*np.sin(ascension),
                - np.sin(perigee)*np.cos(ascension)
                - np.cos(perigee)*np.cos(inclination)*np.sin(ascension),
                np.sin(ascension)*np.sin(inclination)
            ],
            [
                np.cos(perigee)*np.sin(ascension)
                + np.sin(perigee)*np.cos(inclination)*np.cos(ascension),
                - np.sin(perigee)*np.sin(ascension)
                + np.cos(perigee)*np.cos(inclination)*np.cos(ascension),
                - np.cos(ascension)*np.sin(inclination)
            ],
            [
                np.sin(perigee)*np.sin(inclination),
                np.cos(perigee)*np.sin(inclination),
                np.cos(inclination)
            ]
        ])

        inertial = np.dot(matrix_trans, orbital)

        # Calculate the Julian Day Number of the observed year
        A = np.floor((year-1)/100)
        B = np.floor(A/4)
        C = 2 - A + B
        E = np.floor(365.25*((year-1)+4716))
        F = np.floor(30.6001*(12+1))
        JD = C + 31 + E + F - 1524.5

        # Calculate T

        T = (JD-2451545)/36525

        GMST_1 = (
                24110.584841
                + 8640184.812866 * T
                + 0.093104 * T**2
                - 0.000006210 * T**3)/3600

        GMST_1 = np.mod(GMST_1, 24)

        # Calculate GMST at time t

        GMST = GMST_1 + 0.0657098243 * day + 1.00273791 * 24 * current_time_day

        GMST = np.mod(GMST, 24)

        GMST_degree = GMST*15

        GMST_rad = GMST_degree * np.pi/180

        # Greenwich coordinates transformation

        greenwich_matrix_trans = np.array([
            [np.cos(GMST_rad), np.sin(GMST_rad), 0],
            [-np.sin(GMST_rad), np.cos(GMST_rad), 0],
            [0, 0, 1]
        ])

        greenwich = np.dot(greenwich_matrix_trans, inertial)

        # Calculate the geocentric latitude
        # The polar radius of the Earth in km
        R_p = 6356.755

        phi_gc = np.arctan((R_p/R_e)*np.tan(latitude))

        # Calculate the earth radius r_t at the terminal position

        h = elevation * 10**(-3)

        r_t = h + (R_e*R_p)/np.sqrt(
            (R_e*np.sin(phi_gc))**2 + (R_p*np.cos(phi_gc))**2
        )

        # The terminal Cartesian coordinates in the Greenwich system

        x_terminal = r_t * np.cos(longitude) * np.cos(phi_gc)

        y_terminal = r_t * np.sin(longitude) * np.cos(phi_gc)

        z_terminal = r_t * np.sin(phi_gc)

        #  Terminal-satelite vector rho_g

        rho_g_x = greenwich[0] - x_terminal

        rho_g_y = greenwich[1] - y_terminal

        rho_g_z = greenwich[2] - z_terminal

        rho_g = np.array([rho_g_x, rho_g_y, rho_g_z])

        topo_trans = np.array([
            [
                np.sin(phi_gc)*np.cos(longitude),
                np.sin(phi_gc)*np.sin(longitude),
                - np.cos(phi_gc)
            ],
            [
                -np.sin(longitude),
                np.cos(longitude),
                0
            ],
            [
                np.cos(phi_gc) * np.cos(longitude),
                np.cos(phi_gc) * np.sin(longitude),
                np.sin(phi_gc)
            ]
        ])

        rho_h = np.dot(topo_trans, rho_g)

        # Calculate elevation angles, slant paths

        slant_path = np.sqrt(
            (rho_h[0])**2 + (rho_h[1])**2 + (rho_h[2])**2
        )

        elevation_angle = np.arcsin(rho_h[2]/slant_path) * (180/np.pi)

        zenith_angle = 90 - elevation_angle

        return slant_path, zenith_angle

    def pos_in_ECI(
            self, year, day, hour, minute, second, UTC
    ):
        current_time_second = (hour - UTC)*3600 + minute*60 + second

        time_after_epoch = current_time_second - self.epoch_time * 24 * 3600

        current_time_day = self.epoch_time + (time_after_epoch/3600)/24

        inclination = self.inclination * np.pi/180

        perigee = self.perigee * np.pi/180

        ascension = self.ascension * np.pi/180

        n_s = (self.mean_motion * 2 * np.pi)/86400

        mean_anomaly = self.mean_anomaly * np.pi/180 + n_s*time_after_epoch
        mean_anomaly = np.mod(mean_anomaly, 360)

        eccentric_anomaly = (
            mean_anomaly + self.eccentricity*np.sin(mean_anomaly)
            + (self.eccentricity**2/2)*np.sin(2*mean_anomaly)
            + (self.eccentricity**3/8) * (
                3 * np.sin(3*mean_anomaly) - np.sin(mean_anomaly))
        )

        # the graviational constant in Km^3/s^2
        muy = 398600.5

        semi_major_axis = (muy/(n_s**2))**(1/3)

        x_0 = semi_major_axis * (np.cos(eccentric_anomaly) - self.eccentricity)

        y_0 = (
            semi_major_axis
            * np.sqrt(1 - self.eccentricity**2)
            * np.sin(eccentric_anomaly)
        )

        z_0 = 0

        orbital = np.array([x_0, y_0, z_0])

        # Transformation to earth-centered inertial (ECI) coordinates
        matrix_trans = np.array([
            [
                np.cos(perigee)*np.cos(ascension)
                - np.sin(perigee)*np.cos(inclination)*np.sin(ascension),
                - np.sin(perigee)*np.cos(ascension)
                - np.cos(perigee)*np.cos(inclination)*np.sin(ascension),
                np.sin(ascension)*np.sin(inclination)
            ],
            [
                np.cos(perigee)*np.sin(ascension)
                + np.sin(perigee)*np.cos(inclination)*np.cos(ascension),
                - np.sin(perigee)*np.sin(ascension)
                + np.cos(perigee)*np.cos(inclination)*np.cos(ascension),
                - np.cos(ascension)*np.sin(inclination)
            ],
            [
                np.sin(perigee)*np.sin(inclination),
                np.cos(perigee)*np.sin(inclination),
                np.cos(inclination)
            ]
        ])

        inertial = np.dot(matrix_trans, orbital)

        # Calculate the Julian Day Number of the observed year
        A = np.floor((year-1)/100)
        B = np.floor(A/4)
        C = 2 - A + B
        E = np.floor(365.25*((year-1)+4716))
        F = np.floor(30.6001*(12+1))
        JD = C + 31 + E + F - 1524.5

        # Calculate T

        T = (JD-2451545)/36525

        GMST_1 = (
                24110.584841
                + 8640184.812866 * T
                + 0.093104 * T**2
                - 0.000006210 * T**3)/3600

        GMST_1 = np.mod(GMST_1, 24)

        # Calculate GMST at time t

        GMST = GMST_1 + 0.0657098243 * day + 1.00273791 * 24 * current_time_day

        GMST = np.mod(GMST, 24)

        GMST_degree = GMST*15

        GMST_rad = GMST_degree * np.pi/180

        # Latitude of the satellite in ECI
        lambda_sat = np.arcsin(
            inertial[1]
            / (np.sqrt(inertial[0]**2 + inertial[1]**2))
        ) - GMST_rad

        # Longtitude of the satellite in ECI
        phi_sat = np.arctan(
            inertial[2]
            / (np.sqrt(inertial[0]**2 + inertial[1]**2))
        )

        # altitude of the satellite (km)
        h_sat = np.sqrt(
            inertial[0]**2 + inertial[1]**2 + inertial[2]**2
        )

        return lambda_sat, phi_sat, h_sat


def computeGeometricBetweenSats(
        sat1, sat2, year, day, hour, minute, second, UTC
        ):

    lambda_1, phi_1, h_1 = sat1.pos_in_ECI(
        year=year, day=day, hour=hour,
        minute=minute, second=second, UTC=UTC
    )

    lambda_2, phi_2, h_2 = sat2.pos_in_ECI(
        year=year, day=day, hour=hour,
        minute=minute, second=second, UTC=UTC
    )

    r_lc = 2 * h_1 * np.sin(
        0.5 * np.arccos(
            np.sin(phi_1) * np.sin(phi_2)
            + np.cos(phi_1) * np.cos(phi_2) * np.cos(lambda_1 - lambda_2)
            )
        )

    r_mc = h_2 - h_1

    slant_path = np.sqrt(r_lc**2 + r_mc**2)

    elevation_angle_rad = np.arctan(r_mc/r_lc)

    elevation_angle = elevation_angle_rad*180/np.pi

    zenith_angle = 90 - elevation_angle

    return slant_path, zenith_angle


def computeDistanceBetweenTwoPlaces(lat_1, lat_2, lon_1, lon_2):
    """Haversine formula

    Args:
        lat_1 (_type_): _description_
        lat_2 (_type_): _description_
        lon_1 (_type_): _description_
        lon_2 (_type_): _description_
    """

    lat_1_rad = lat_1 * np.pi/180

    lat_2_rad = lat_2 * np.pi/180

    lon_1_rad = lon_1 * np.pi/180

    lon_2_rad = lon_2 * np.pi/180

    dLat = lat_2_rad - lat_1_rad

    dLon = lon_2_rad - lon_1_rad

    a = (
        np.sin(dLat/2)**2 + np.cos(lat_1_rad)
        * np.cos(lat_2_rad) * np.sin(dLon/2)**2
    )

    r = 6371000

    d = 2 * r * np.arcsin(np.sqrt(a))

    return d
