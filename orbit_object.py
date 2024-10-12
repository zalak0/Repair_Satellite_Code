import numpy as np

class Orbit:
    """
    An object defining a single orbit
    """
    def __init__(self: object, orbit: tuple[float], mu: float):
        """
        Function for orbit definition, calculates relevant orbital parameters

        Args:
            self (object): The orbit object where the parameters are stored
            orbit (tuple[float]): an input list of (perigee_radius, apogee_radius,
            inclination angle, RAAN, argument of perigee and starting true anomaly)
            mu (float): Gravitational parameter = 398600 [km^3/s^2]
        """
        self.r_p = orbit[0]                 # Radius of perigee [km]
        self.r_a = orbit[1]                 # Radius of apogee [km]
        self.i = np.deg2rad(orbit[2])       # Inclination angle [rad]
        self.raan = np.deg2rad(orbit[3])    # RAAN angle [rad]
        self.omega = np.deg2rad(orbit[4])   # Argument of perigee [rad]
        self.starting_true_anomaly = np.deg2rad(orbit[5])   # True anomaly of satellite at time = 0 [rad]

        self.h = np.sqrt(2*mu*self.r_a*self.r_p/(self.r_a+self.r_p))    # Specific angular momentum [km^2/s]
        self.e = (self.r_a-self.r_p)/(self.r_a+self.r_p)                # Eccentricity
        self.T = 2*np.pi/np.sqrt(mu) * ((self.r_a + self.r_p)/2)**(3/2) # Period [s]

        self.Q = self.transform_perifocal_eci() # Transformation matrix from specific orbit perifocal frame to the ECI frame

    def transform_perifocal_eci(self: object) -> tuple[tuple[float]]:
        """
        Generates the transformation matrix that can transform any vector from the perifocal frame
        specific to this orbit into a general ECI frame

        Returns:
            tuple[tuple[float]]: The transformation matrix
        """
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
    
    def theta_from_t(self: object, time: float, mu: float) -> float:
        """
        Generates the true anomaly position of a satellite after some time t has passed
        Uses Newton's method of integrating to find the eccentric anomaly 

        Args:
            self (object): Orbit object
            time (float): Current time
            mu (float): Gravitational parameter = 398600 [km^3/s^2]

        Returns:
            float: Satellite true anomaly at its current location
        """
        time = time % self.T            # Time since perigee [s]
        
        M_e = mu**2 / self.h**3 * (1-self.e**2)**(3/2) * time  # Kepler's Equation for the Mean Anomaly

        # Newton's Method of Integration to find the Eccentric ANomaly
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

        true_anomaly = 2*np.atan(np.sqrt((1+self.e)/(1-self.e)) * np.tan(E/2)) # Current true_anomaly

        return true_anomaly
    
    def t_from_theta(self: object, theta: float, mu: float) -> float:
        """
        Generates the time since perigee for some true anomaly position
        Usese Kepler's mean anomaly and eccentric anomaly definitions

        Args:
            self (object): Orbit object
            theta (float): Current true anomaly position
            mu (float): Gravitational parameter = 398600 [km^3/s^2]

        Returns:
            float: Satellite time since perigee 
        """
        E = 2*np.atan(np.sqrt((1-self.e)/(1+self.e)) * np.tan(theta/2)) # Eccentric anomaly 

        M_e = E - self.e*np.sin(E)  # Mean anomaly

        time = self.h**3 / mu**2 / (1-self.e**2)**(3/2) * M_e   # Current time since perigee

        return time