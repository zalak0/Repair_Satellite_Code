import numpy as np

def mean_to_true_anomaly(mean_anomaly : float, eccentricity : float):
    # Convert mean anomaly to eccentric anomaly using iterative method
    E = mean_anomaly - eccentricity/2  # Initial guess
    tolerance = 1e-6
    while True:
        f = E - eccentricity * np.sin(E) - mean_anomaly
        f_dash = 1 - eccentricity * np.cos(E)
        delta = f / f_dash
        E -= delta
        if abs(delta) < tolerance:
            break

    # Convert eccentric anomaly to true anomaly
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + eccentricity) *
                                    np.sin(E / 2), np.sqrt(1 - eccentricity) * np.cos(E / 2))

    return true_anomaly

def period(semimajor_axis : float, mu : float):
    T = 2 * np.pi/np.sqrt(mu) * semimajor_axis **(3/2)
    return T

def semimajor_reversed(period : float, mu :float):
    a = ((period * np.sqrt(mu))/(2 * np.pi))**(2/3)
    return a

def eccentricity(perigee : float, apogee : float) -> float:
    e = (apogee - perigee)/(perigee + apogee)
    return e

def eccentricity_circ(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """Calculate the eccentricity of an orbit.

    Args:
        r (np.ndarray): Position vector
        v (np.ndarray): Velocity vector
        mu (float): Gravitational parameter

    Returns:
        float: The eccentricity of the orbit
    """
    # TODO: Implement the eccentricity calculation
    h = np.cross(r, v)
    r_mag = np.linalg.norm(r)
    e = (np.cross(v, h))/mu - r/r_mag
    return np.linalg.norm(e)

def angular_momentum(perigee : float, apogee : float, mu : float) -> float:
    h = np.sqrt(2 * mu) * np.sqrt((perigee * apogee)/(perigee + apogee))
    return h

def delta_v(angular_momentum_1, angular_momentum_2, radius) -> float:
    v1 = angular_momentum_1/radius
    v2 = angular_momentum_2/radius

    delta_v = abs(v2 - v1)
    return delta_v

def delta_plane(angular_momentum : float, radius : float, ang_1 : float, ang_2 :float, print_stuff: int = 0) -> float:
    v = angular_momentum/radius
    delta_ang = abs(ang_1 - ang_2)
    if print_stuff:
        print(f"Angular momentum:        {angular_momentum}")
        print(f"Radius:                  {radius}")
        print(f"Velocity at RAAN change: {v}")
        print(f"Angle of change:         {np.degrees(delta_ang)}")
    delta_v = 2 * v * np.sin(delta_ang/2)

    return delta_v

def delta_comb_plane(angular_momentum : float, radius : float, inc_1 : float, inc_2 :float,
                raan_1 : float, raan_2 : float, print_stuff: int = 0) -> float:
    v = angular_momentum/radius
    delta_ang = np.arccos(np.cos(abs(raan_1-raan_2)) * np.sin(inc_1) * np.sin(inc_2) + \
                    np.cos(inc_1) * np.cos(inc_2))
    if print_stuff:
        print(f"Angular momentum:        {angular_momentum}")
        print(f"Radius:                  {radius}")
        print(f"Velocity at RAAN change: {v}")
        print(f"Angle of change:         {np.degrees(delta_ang)}")
    delta_v = 2 * v * np.sin(delta_ang/2)

    return delta_v

def change_in_mass(delta_v, m0, specific_impulse, gravity = 9.81) -> float:
    delta_vms = delta_v * 1000
    dm = m0 * (1 - np.e**(-delta_vms/(specific_impulse * gravity)))
    mf = m0 - dm
    return dm, mf

def total_time(period_mid : float, period_rise : float,
               i_diff : float, points_sim : float, T_return: int = 0 ):
    (i_diff_hohmann, i_diff_rise, i_diff_shift) = i_diff
    dt_hohmann = period_mid/points_sim
    dt_rised = period_rise/points_sim
    time_hohmann = dt_hohmann * i_diff_hohmann
    time_rise =  dt_rised * i_diff_rise
    time_shift =  dt_rised * i_diff_shift
    total_time = time_hohmann + time_rise + time_shift

    if not T_return:
        print(f"Time taken to rise orbit (hohmann):             {time_hohmann:.3f}s")
        print(f"Time taken for plane combo (plane):             {time_rise:.3f}s")
        print(f"Time taken to lower orbit(hohmann plane):       {time_shift:.3f}s")
        print(f"Total time (to reach target satellite orbit):   {total_time:.3f}s")
        return total_time
    else:
        return total_time