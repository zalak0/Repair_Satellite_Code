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

def angular_momentum(perigee : float, apogee : float, mu : float) -> float:
    h = np.sqrt(2 * mu) * np.sqrt((perigee * apogee)/(perigee + apogee))
    return h

def delta_v(angular_momentum_1, angular_momentum_2, radius) -> float:
    v1 = angular_momentum_1/radius
    v2 = angular_momentum_2/radius

    delta_v = abs(v2 - v1)
    return delta_v

def delta_plane(angular_momentum : float, radius : float, ang_1 : float, ang_2 :float) -> float:
    v = angular_momentum/radius
    delta_ang = abs(ang_1 - ang_2)
    delta_v = 2 * v * np.sin(delta_ang/2)

    return delta_v

def change_in_mass(delta_v, mf, specific_impulse, gravity = 9.81) -> float:
    delta_vms = delta_v * 1000
    m0 = np.e**(delta_vms/(specific_impulse*gravity)) * mf
    dm = m0 - mf
    print(f"Fuel mass required (Isp = {specific_impulse}):              {m0:.3f}kg")
    return dm

def total_time(period_chase : float, period_mid : float,
               i_diff : float, points_sim : float, T_return: int = 0 ):
    i_diff_inc, i_diff_raan, i_diff_hohmann = i_diff
    dt_inc_raan = period_chase/points_sim
    dt_hohmann = period_mid/points_sim
    time_inc = dt_inc_raan * i_diff_inc
    time_raan = dt_inc_raan * i_diff_raan
    time_hohmann = dt_hohmann * i_diff_hohmann
    total_time = time_inc + time_raan + time_hohmann

    if not T_return:
        print(f"Time taken for inclination change:                 {time_inc:.3f}s")
        print(f"Time taken for RAAN change:                        {time_raan:.3f}s")
        print(f"Time taken for hohmann transfer:                   {time_hohmann:.3f}s")
        print(f"Total time (to reach target satellite orbit):      {total_time:.3f}s")
    else:
        return total_time