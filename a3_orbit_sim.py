"""
a2_simulation.py

File repsonsible for simulating and extracting array values, mainly used
for plotting purposes.
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.integrate import solve_ivp
from orbit_object import Orbit as orb_obj
import a3_long_calc as lc
import a3_formulae as form

# Define a global flag to check if values have been printed
printed = False

def orbital_derivatives(t : tuple[float], y : tuple[float], period : float) -> np.ndarray:
    """Differential/defining function of the satellite, calculating the derivative of
    right angle of ascension, argument of perigee, and mean anomaly using numerical
    analysis (solve_ivp)

    Args:
        t (tuple[float]): Time of simulation
        y (tuple[float]): Array of current right angle of ascension
                        argument of perigee and mean anomaly
        eccentricity (float)
        semimajor_axis (float)
        inc_ang (float): inclination angle
        jtwo (float): J2 pertubation constant
        period (float)
        earth_rad (float): Radius of earth
        mu (float): Gravitational constant
        string (str): String identifying if the orbit takes into account J2 or not

    Returns:
        np.ndarray: Array of differential values to update y
    """
    raan_dot = 0
    omega_dot = 0

    # Rates of change of the orbital elements
    draan_dt = raan_dot
    domega_dt = omega_dot
    dmean_dt = 2 * np.pi / period

    return [draan_dt, domega_dt, dmean_dt]


def sim_orbit(cur_orb, mean_anomaly, mu : float, points_sim : tuple[float]
              ) -> tuple[np.ndarray, np.ndarray,
                        np.ndarray, np.ndarray, np.ndarray]:

    """Simulates the ground-track of a satellite in semi-ideal conditions (without account of J2)

    Args:
        eccentricity (float)
        semimajor_axis (float)
        inc_ang (float): inclination angle
        raan (float): Right angle of ascension
        omega (float): Argument of perigee (degrees)
        mean_anomaly (float)
        period (float)
        omega_e (float): Rotational constant of Earth (rad/s)
        mu (float): Gravitational constant
        time_sim (tuple[float]): Simulation time (s)
        points_sim (tuple[float]): Frequency of calculation within simulation

    Returns:
        tuple[np.ndarray, np.ndarray]: latitude and longitude arrays to plot
        aswell as x,y and z components of ECI orbit
    """
    print("Simulating " + cur_orb.name + "...")

    # The time the satellite has covered w.r.t perigee
    t0 = np.radians(mean_anomaly)/(2*np.pi) * cur_orb.T

    # Initial vector for numerical analysis
    y0 = [np.radians(cur_orb.raan), np.radians(cur_orb.omega), np.radians(mean_anomaly)]

    # The time span of the simulation
    t_span = [t0, t0 + cur_orb.T]

    # The amount of points that will be calculated within the
    # respected time span.
    t_eval = np.linspace(t_span[0], t_span[1], points_sim)

    # Solve the differential equations
    sol = solve_ivp(orbital_derivatives, t_span, y0, args=(cur_orb.T,), t_eval=t_eval)

    # Extract results
    raan_sol, omega_sol, mean_anomaly_sol = sol.y

    # Simulate the latitude and longitude of the satellite (to represent ground-track)
    x_eci, y_eci, z_eci, v_eci =  sim_orbit_values(cur_orb, t_eval, mean_anomaly_sol, mu)
    return (x_eci, y_eci, z_eci), v_eci

def sim_orbit_values(cur_orb, t_eval : tuple[float], mean_anomaly_sol : tuple[float], mu : float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """With the change in significant parameters known, this function calculates the
    the arrays for the ground-track of the satellite and the component arrays of the
    ECI orbit

    Args:
        eccentricity (float)
        semimajor_axis (float)
        inc_ang (float): inclination angle
        t_eval (tuple[float]): Frequency of calculations within time span
        raan_sol (tuple[float]): Change in right angle of ascension
                                for every point of calculation
        omega_sol (tuple[float]): Change in argument of perigee
                                for every point of calculation
        mean_anomaly_sol (tuple[float]): Change in mean anomaly
                                        for every point of calculation
        omega_e (float): Rotational constant of Earth (rad/s)
        mu (float): Gravitational constant

    Returns:
        longitude and latitude arrays and x,y and z components of ECI orbit
    """

    # Initialise arrays for plotting
    x_eci = np.zeros(len(t_eval))
    y_eci = np.zeros(len(t_eval))
    z_eci = np.zeros(len(t_eval))
    v_eci = np.zeros((len(t_eval), 3))

    # Simulate the latitude and longitude of the satellite (to represent ground-track)
    for i in range(len(t_eval)):
        # Find true anomaly, since mean anomaly changes due to J2
        theta = form.mean_to_true_anomaly(mean_anomaly_sol[i], cur_orb.e)

        # Find radius of satellite in perifocal frame then switch to ECI frame
        r_pff, v_pff = cur_orb.elements_to_perifocal(theta, mu)
        q = cur_orb.transform_perifocal_eci()
        r_eci = q @ r_pff
        v_eci[i, :] = q @ v_pff

        array_2d = np.array(r_eci)

        # Convert to 1D array (3D vector)
        r_eci_fix = array_2d.flatten()  # or use array_2d.ravel()`

        x_eci[i] = r_eci_fix[0]
        y_eci[i] = r_eci_fix[1]
        z_eci[i] = r_eci_fix[2]

    return x_eci, y_eci, z_eci, v_eci

def sim_delta_time(current : tuple, target : tuple, omega_e : float,
                   points_sim : float,  mu : float) -> tuple[float]:

    cur_orb = orb_obj(current, mu)
    targ_orb = orb_obj(target, mu)

    eccentricity_mid = form.eccentricity(cur_orb.r_p, targ_orb.r_a)
    semimajor_mid = (cur_orb.r_p + targ_orb.r_a)/2
    period_mid = form.period(semimajor_mid, mu)
    h_mid = form.angular_momentum(cur_orb.r_p, targ_orb.r_a, mu)

    # Assume orbit starts at perigee, we will fix this later
    mean_anomaly = 0
    arg_perigee_targ = 0

    # Use Question 1 functions for simplicity and modularity
    # Note that variables starting with ground are not used

    chase_orb = orb_obj(("Chase orbit", cur_orb.r_p, cur_orb.r_a, cur_orb.T,
                        cur_orb.h, cur_orb.i, cur_orb.raan, targ_orb.omega), mu)
    orb_chase, vel_chase = sim_orbit(chase_orb, mean_anomaly, mu,
                                   points_sim)

    inc_orb = orb_obj(("Inclination orbit", cur_orb.r_p, cur_orb.r_a, cur_orb.T,
                        cur_orb.h, targ_orb.i, cur_orb.raan, cur_orb.omega), mu)
    orb_inc, vel_inc = sim_orbit(inc_orb, mean_anomaly, mu,
                                   points_sim)

    raan_orb = orb_obj(("RAAN orbit", cur_orb.r_p, cur_orb.r_a, cur_orb.T,
                        cur_orb.h, targ_orb.i, targ_orb.raan, targ_orb.omega), mu)
    orb_raan, vel_raan = sim_orbit(raan_orb, mean_anomaly, mu,
                                   points_sim)

    hohmann_orb = orb_obj(("Hohmann orbit", cur_orb.r_p, targ_orb.r_a, period_mid,
                        h_mid, targ_orb.i, targ_orb.raan, targ_orb.omega), mu)
    orb_hohmann, vel_hohmann = sim_orbit(hohmann_orb, mean_anomaly, mu,
                                   points_sim)

    final_orb = orb_obj(("Target orbit", targ_orb.r_p, targ_orb.r_a, targ_orb.T,
                        targ_orb.h, targ_orb.i, targ_orb.raan, targ_orb.omega), mu)
    orb_targ, vel_targ = sim_orbit(final_orb, mean_anomaly, mu,
                                   points_sim)

    # Find the intersection between all the graphs
    # This is because the transfers occur only at these intersections
    # and we want to start and finish the orbit at specific points
    # Which are the points where the delta v occurs

    print("Calculating orbit intersections...")
    r_inc_start1, r_inc_start2 = lc.check_intersection(orb_chase, orb_inc)
    r_raan_start1, r_raan_start2 = lc.check_intersection(orb_inc, orb_raan)
    r_hohmann_start1, r_hohmann_start2 = lc.check_intersection(orb_raan, orb_hohmann)
    r_hohmann_finish1, r_hohmann_finish2 = lc.check_intersection(orb_hohmann, orb_targ)

    print("Fixing up orbits...")
    # Now we know orbit intersections, lets fix our orbits up
    x_inc_fix, y_inc_fix, z_inc_fix, i_diff_inc = \
                    lc.fix_orbit(orb_inc, r_inc_start1, r_raan_start2, 0)
    x_raan_fix, y_raan_fix, z_raan_fix, i_diff_raan = \
                    lc.fix_orbit(orb_raan, r_raan_start2, r_hohmann_start1, 0)
    x_hohmann_fix, y_hohmann_fix, z_hohmann_fix, i_diff_hohmann = \
                    lc.fix_orbit(orb_hohmann, r_hohmann_start1, r_hohmann_finish1, 0)

    return (i_diff_inc, i_diff_raan, i_diff_hohmann), period_mid