import numpy as np
from orbit_object import Orbit

def intersection_orbit(o0: tuple[float], o2: tuple[float], mu: float) -> tuple[float]:
    # Finds the true anomaly of orbit 2 at it's point of intersection with the original orbit
    Q20 = np.array(np.matmul(np.transpose(o0.Q), o2.Q))         # Transformation vector from Orbit 1 to Orbit 2

    theta_2_1 = np.atan(- Q20[2][0] / Q20[2][1])                # True Anomally where z-axis = 0 
    theta_2_2 = theta_2_1 + np.pi                               # Second true anomaly value
    r_2_1 = np.array(np.matmul(o2.Q, o2.h**2/mu/(1 + o2.e*np.cos(theta_2_1)) * np.array([np.cos(theta_2_1), np.sin(theta_2_1), 0])))[0]   # Radius 1
    r_2_2 = np.array(np.matmul(o2.Q, o2.h**2/mu/(1 + o2.e*np.cos(theta_2_2)) * np.array([np.cos(theta_2_2), np.sin(theta_2_2), 0])))[0]   # Radius 2

    if np.linalg.norm(r_2_1) > np.linalg.norm(r_2_2):
        u_r_0 = np.array(np.transpose(o0.Q), np.matmul(r_2_1/np.linalg.norm(r_2_1)))[0]     # Unit vector towards intersection in plane 0
        theta_0 = np.pi + np.acos(u_r_0[0])                                                 # True Anomaly corresponding to unit vector
        r_0 = o0.h**2/mu/(1 + o0.e*np.cos(theta_0)) * np.array([np.cos(theta_0), np.sin(theta_0), 0])

        theta_n = np.atan(- o0.Q[2][0] / o0.Q[2][1]) 
        n = np.array([np.cos(theta_n), np.sin(theta_n), 0])                                 # Unit vector in direction of ascending node
        omega = np.dot(n, r_0)/(np.linalg.norm(n)*np.linalg.norm(r_0))

        return theta_0, theta_2_1, np.linalg.norm(r_0), np.linalg.norm(r_2_1), omega
    else:
        u_r_0 = np.array(np.matmul(r_2_2/np.linalg.norm(r_2_2), np.transpose(o0.Q)))[0]     # Unit vector in direction of apse line
        theta_0 = np.pi + np.acos(u_r_0[0])
        r_0 = o0.h**2/mu/(1 + o0.e*np.cos(theta_0)) * np.array([np.cos(theta_0), np.sin(theta_0), 0])

        theta_n = np.atan(- np.array(o0.Q)[2][0] / np.array(o0.Q)[2][1])            
        n = np.array([np.cos(theta_n), np.sin(theta_n), 0])                         # Unit vector in direction of ascending node
        omega = np.dot(n, r_0)/(np.linalg.norm(n)*np.linalg.norm(r_0))

        return theta_0, theta_2_2, np.linalg.norm(r_0), np.linalg.norm(r_2_1), omega

def transfer_delta_v(oi: tuple[float], of: tuple[float], ta_i: float, ta_f: float, mu: float) -> float:
    v_i = mu/oi.h * np.array([-np.sin(ta_i), oi.e + np.cos(ta_i), 0])
    vec_v_i = np.array(np.matmul(oi.Q, v_i))[0]
    v_f = mu/of.h * np.array([-np.sin(ta_f), of.e + np.cos(ta_f), 0])
    vec_v_f = np.array(np.matmul(of.Q, v_f))[0]
    return np.abs(np.linalg.norm(vec_v_f - vec_v_i))

def transfer(
    orbit_0: tuple[float],
    orbit_2: tuple[float],
    t: float
) -> tuple[float]: # Transfers between any two orbits
    mu = 398600     # Earth gravitational parameter (km^3/s^2)

    o0 = Orbit(orbit_0, mu)
    o2 = Orbit(orbit_2, mu)

    theta_0, theta_2, r_a, r_p, omega = intersection_orbit(o0, o2, mu)
    
    o1 = Orbit([r_a, r_p, o0.i, o0.raan, omega, 0], mu)

    dv1 = transfer_delta_v(o0, o1, theta_0, 0, mu)
    dv2 = transfer_delta_v(o1, o2, np.pi, theta_2, mu)

    dv = dv1 + dv2

    t1 = o0.t_from_theta(theta_0, mu) - o0.t_from_theta(o0.theta, mu)   # Time taken to go from starting ta to first maneuver ta (NEEDS WORK)
    t2 = o1.T/2 # Time taken to travel to point of second maneuver
    t3 = 0 # Time taken for phasing maneuver (TBD)


    t = t + t1 + t2 + t3

    return dv, t