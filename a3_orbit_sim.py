"""
a2_simulation.py

File repsonsible for simulating and extracting array values, mainly used
for plotting purposes.
"""
from scipy.integrate import solve_ivp
from orbit_object import Orbit as orb_obj
from scipy import constants as spconst
import numpy as np
import matplotlib.pyplot as plt
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
        r_eci = cur_orb.Q @ r_pff
        v_eci[i, :] = cur_orb.Q @ v_pff

        array_2d = np.array(r_eci)

        # Convert to 1D array (3D vector)
        r_eci_fix = array_2d.flatten()  # or use array_2d.ravel()`

        x_eci[i] = r_eci_fix[0]
        y_eci[i] = r_eci_fix[1]
        z_eci[i] = r_eci_fix[2]

    return x_eci, y_eci, z_eci, v_eci

def sim_delta_time(current : tuple, target : tuple,
                   points_sim : float,  mu : float) -> tuple[float]:

    cur_orb = orb_obj(current, mu)
    targ_orb = orb_obj(target, mu)

    semimajor_mid = (cur_orb.r_p + targ_orb.r_a)/2
    period_mid = form.period(semimajor_mid, mu)
    h_mid = form.angular_momentum(cur_orb.r_p, targ_orb.r_a, mu)

    semimajor_rise = (targ_orb.r_a + targ_orb.r_a)/2
    period_rise = form.period(semimajor_rise, mu)
    h_rise = form.angular_momentum(targ_orb.r_a, targ_orb.r_a, mu)

    # Assume orbit starts at perigee, we will fix this later
    mean_anomaly = 0
    arg_perigee_targ = 0

    # Use Question 1 functions for simplicity and modularity
    # Note that variables starting with ground are not used

    chase_orb = orb_obj(("Chase orbit", cur_orb.r_p, cur_orb.r_a, cur_orb.T,
                        cur_orb.h, np.degrees(cur_orb.i), np.degrees(cur_orb.raan),
                        np.degrees(cur_orb.omega)), mu)
    orb_chase, vel_chase = sim_orbit(chase_orb, mean_anomaly, mu,
                                   points_sim)

    hohmann_orb = orb_obj(("Hohmann orbit", cur_orb.r_p, targ_orb.r_a, period_mid,
                        h_mid, np.degrees(cur_orb.i), np.degrees(cur_orb.raan),
                        np.degrees(cur_orb.omega)), mu)
    orb_hohmann, vel_hohmann = sim_orbit(hohmann_orb, mean_anomaly, mu,
                                   points_sim)

    rise_orb = orb_obj(("Rised orbit", targ_orb.r_a, targ_orb.r_a, period_rise,
                        h_rise, np.degrees(cur_orb.i), np.degrees(cur_orb.raan),
                        np.degrees(cur_orb.omega)), mu)
    orb_rise, vel_rise = sim_orbit(rise_orb, mean_anomaly, mu,
                                   points_sim)

    shift_orb = orb_obj(("Shifted orbit", targ_orb.r_a, targ_orb.r_a, period_rise,
                        h_rise, np.degrees(targ_orb.i), np.degrees(targ_orb.raan),
                        np.degrees(cur_orb.omega)), mu)
    orb_shift, vel_shift = sim_orbit(shift_orb, mean_anomaly, mu,
                                   points_sim)

    final_orb = orb_obj(("Target orbit", targ_orb.r_p, targ_orb.r_a, targ_orb.T,
                        targ_orb.h, np.degrees(targ_orb.i), np.degrees(targ_orb.raan),
                        np.degrees(targ_orb.omega)), mu)
    orb_targ, vel_targ = sim_orbit(final_orb, mean_anomaly, mu,
                                   points_sim)

    # Find the intersection between all the graphs
    # This is because the transfers occur only at these intersections
    # and we want to start and finish the orbit at specific points
    # Which are the points where the delta v occurs

    print("Calculating orbit intersections...")
    r_rise_start1, r_rise_start2 = lc.check_intersection(orb_chase, orb_hohmann)
    r_rise_fin1, r_rise_fin2 = lc.check_intersection(orb_hohmann, orb_rise)
    r_shift_start1, r_shift_start2 = lc.check_intersection(orb_rise, orb_shift)
    r_fin_start1, r_fin_start2 = lc.check_intersection(orb_shift, orb_targ)

    print("Fixing up orbits...")
    # Now we know orbit intersections, lets fix our orbits up
    x_hohmann_fix, y_hohmann_fix, z_hohmann_fix, i_diff_hohmann = \
                    lc.fix_orbit(orb_hohmann, r_rise_start1, r_rise_fin1, 0)
    x_rise_fix, y_rise_fix, z_rise_fix, i_diff_rise = \
                    lc.fix_orbit(orb_rise, r_rise_fin1, r_shift_start1, 0)
    x_shift_fix, y_shift_fix, z_shift_fix, i_diff_shift = \
                    lc.fix_orbit(orb_shift, r_shift_start1, r_fin_start1, 0)

    # # Initialise plot and plot each graph
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # # Plot the chase orbit
    # x_chase, y_chase, z_chase = orb_chase
    # ax.plot(x_chase, y_chase, z_chase, label='Chase Orbit', color='r')

    # # Plot the target orbit
    # x_targ, y_targ, z_targ = orb_targ
    # ax.plot(x_targ, y_targ, z_targ, label='Target Orbit', color='orange')

    # x_shift, y_shift, z_shift = orb_shift
    # x_rise, y_rise, z_rise = orb_rise
    # # Plotting rise orbit fix (x, y, z)
    # ax.plot(x_hohmann_fix, y_hohmann_fix, z_hohmann_fix, label='Hohmann Orbit', color='b')
    # ax.plot(x_rise_fix, y_rise_fix, z_rise_fix, label='Rise Orbit', color='g')
    # ax.plot(x_shift_fix, y_shift_fix, z_shift_fix, label='Shift Orbit',)

    # # Plot Earth
    # # Create a grid of points in spherical coordinates
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)

    # # Convert spherical coordinates to Cartesian coordinates
    # x = 6371 * np.outer(np.cos(u), np.sin(v))
    # y = 6371 * np.outer(np.sin(u), np.sin(v))
    # z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))

    # # Plot the surface of the sphere (the Earth)
    # ax.plot_surface(x, y, z, color='b', rstride=4, cstride=4, alpha=0.4)

    # # Adding labels
    # ax.set_xlabel('X Coordinate (km)')
    # ax.set_ylabel('Y Coordinate (km)')
    # ax.set_zlabel('Z Coordinate (km)')
    # ax.set_title('3D Orbit Visualization')

    # # Add a legend
    # ax.legend()

    # # Set aspect ratio to equal for better visualization
    # ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

    # # Show the plot
    # plt.show()
    return (i_diff_hohmann, i_diff_rise, i_diff_shift), period_mid, period_rise