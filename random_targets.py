import numpy as np

def random_targets(
        n: float # Number of target orbits
        ) -> tuple[tuple[float]]:
    R_E = 6378.1                                # Earth radius (km)
    parking_orbit = [160 + R_E, 160 + R_E, 10, 10, 10, 0]     # Starting Parking Orbit

    targets = [parking_orbit]
    for i in range(n):
        r_p = np.random.randint(160,2000)                   # Random LEO Perigee Radius [km]
        r_a = np.random.randint(r_p, 2000)                  # Random LEO Apogee Radius (with perigee as lower bound) [km]
        i, RAAN, omega = np.random.randint(0, 90, size=3)   # Random inclination, RAAN and argument of perigee angles [deg]
        theta = np.random.randint(0, 360)                   # Satellite true anomaly at t=0
        targets.append([r_p + R_E, r_a + R_E, i, RAAN, omega, theta])

    return targets