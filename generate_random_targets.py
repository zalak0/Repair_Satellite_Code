import numpy as np

def random_targets(
        n: float
        ) -> tuple[tuple[float]]:
    """
    Generates a list of n target orbits to visit using the transfer alogirthm

    Args:
        n (float): The number of target orbits required

    Returns:
        tuple[tuple[float]]: A list of n target orbits in format (perigee radius, apogee radius,
        inclination angle, RAAN, argument of perigee, and true anomaly of satellite at time = 0)
    """
    R_E = 6378.1                                            # Earth radius (km)
    parking_orbit = [160 + R_E, 160 + R_E, 10, 10, 10, 0]   # Starting LEO Parking Orbit

    targets = [parking_orbit]                               # List of targets starting with the parking orbit
    for i in range(n):
        perigee_radius = np.random.randint(160,2000)            # Random LEO Perigee Radius [km]
        apogee_radius = np.random.randint(perigee_radius, 2000) # Random LEO Apogee Radius (with perigee as lower bound) [km]
        i = np.random.randint(0, 90)                            # Random inclination angle [deg]
        RAAN = np.random.randint(0, 90)                         # Random RAAN angle [deg]
        omega = np.random.randint(0, 90)                        # Random argument of perigee angle [deg]
        true_anomaly = np.random.randint(0, 360)                # Satellite true anomaly at time = 0
        targets.append([perigee_radius + R_E, apogee_radius + R_E, i, RAAN, omega, true_anomaly])   # Append orbit to list of orbits

    return targets