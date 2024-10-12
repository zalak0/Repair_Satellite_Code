import numpy as np

class Orbit:
    def __init__(self, orbit : tuple, mu : float):
        self.name = orbit[0]
        self.r_p = orbit[1]
        self.r_a = orbit[2]
        self.T = orbit[3]
        self.h = orbit[4]
        self.i = np.deg2rad(orbit[5])
        self.raan = np.deg2rad(orbit[6])
        self.omega = np.deg2rad(orbit[7])

        self.e = (self.r_a-self.r_p)/(self.r_a+self.r_p)
        self.a = (self.r_a+self.r_p)/2

        self.Q = self.transform_perifocal_eci()

    def elements_to_perifocal(self, theta : float, mu : float) -> tuple[np.ndarray]:
        """Converts a set of orbital elements to the position and
        velocity vectors in the perifocal frame.

        Args:
            h (float): angular momentum
            e (float): eccentricity
            theta (float): true anomaly (radians)
            mu (float): gravitational parameter of the central body

        Returns:
            tuple[np.ndarray]: position and velocity vectors in the perifocal frame
        """
        # Position vector in the perifocal frame
        r_p = (self.h**2/mu * 1/(1 + self.e * np.cos(theta))) * \
                np.array([np.cos(theta), np.sin(theta), 0],
                        dtype = np.float64)
        v_p = mu/self.h * np.array([-np.sin(theta), self.e + np.cos(theta), 0],
                        dtype = np.float64)
        return r_p, v_p

    def transform_perifocal_eci(self):
        M11 = -np.sin(self.raan)*np.cos(self.i)*np.sin(self.omega) + np.cos(self.raan)*np.cos(self.omega)
        M12 = -np.sin(self.raan)*np.cos(self.i)*np.cos(self.omega) - np.cos(self.raan)*np.sin(self.omega)
        M13 = np.sin(self.raan)*np.sin(self.i)

        M21 = np.cos(self.raan)*np.cos(self.i)*np.sin(self.omega) + np.sin(self.raan)*np.cos(self.omega)
        M22 = np.cos(self.raan)*np.cos(self.i)*np.cos(self.omega) - np.sin(self.raan)*np.sin(self.omega)
        M23 = -np.cos(self.raan)*np.sin(self.i)

        M31 = np.sin(self.i)*np.sin(self.omega)
        M32 = np.sin(self.i)*np.cos(self.omega)
        M33 = np.cos(self.i)

        return np.asmatrix([
            [M11, M12, M13],
            [M21, M22, M23],
            [M31, M32, M33]
        ])
