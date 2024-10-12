<<<<<<< HEAD
import numpy as np

class Orbit:
    def __init__(self, orbit, mu):
        self.r_p = orbit[0]
        self.r_a = orbit[1]
        self.i = np.deg2rad(orbit[2])
        self.raan = np.deg2rad(orbit[3])
        self.omega = np.deg2rad(orbit[4])
        self.theta = np.deg2rad(orbit[5])

        self.h = np.sqrt(2*mu*self.r_a*self.r_p/(self.r_a+self.r_p))
        self.e = (self.r_a-self.r_p)/(self.r_a+self.r_p)
        self.T = 2*np.pi/np.sqrt(mu) * ((self.r_a + self.r_p)/2)**(3/2)

        self.Q = self.transform_perifocal_eci()

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

    def theta_from_t(self, t, mu):
        M_e = mu**2 / self.h**3 * (1-self.e**2)**(3/2) * t

        if M_e < np.pi:
            E = M_e + self.e/2
        else:
            E = M_e - self.e/2
        tol = 10**(-8)
        ratio = 1
        while ratio > tol:
            f = E - self.e*np.sin(E) - M_e
            df = 1 - self.e*np.cos(E)
            ratio = f/df
            E = E - ratio

        theta = 2*np.atan(np.sqrt((1+self.e)/(1-self.e)) * np.tan(E/2))

        return theta

    def t_from_theta(self, theta, mu):
        E = 2*np.atan(np.sqrt((1-self.e)/(1+self.e)) * np.tan(theta/2))

        M_e = E - self.e*np.sin(E)

        t = self.h**3 / mu**2 / (1-self.e**2)**(3/2)

        return t