"""
a2_long_calc.py

This files contains certain calculations that require more than 1-2 lines of working out,
or calculating multiple values at once
"""
from scipy import constants as spconst
from scipy import integrate
import numpy as np
import a3_formulae as form
import a3_orbit_sim as orb_sim
import a3_phase_sim as phase_sim

def deduce_tle(file_name : str):
    """Function used to deduce TLE information when provided in a text-file
    """

    with open(file_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()

        # We only need the information in the second-line
        # As those parameters are the ones we learn in this course
        # And the only ones useful for calculation

        # Process Line 1 of TLE
        #line1 = lines[0].strip()
        #line1_split = line1.split()

        # Process Line 2 of TLE
        line2 = lines[1].strip()
        line2_split = line2.split()  # Split line by spaces

        # Line 2 extraction
        # Satellite catalog number
        # sat_cat_num = int(line2_split[1])
        # Inclination angle [degrees]
        inclination = float(line2_split[2])
        # Right Ascension of Ascending Node (RAAN) [degrees]
        raan = float(line2_split[3])
        # Eccentricity (Assumed leading decimal)
        eccentricity = float('0.' + line2_split[4])
        # Argument of Perigee [degrees]
        arg_perigee = float(line2_split[5])
        # Mean Anomaly [degrees]
        mean_anomaly = float(line2_split[6])
        # Mean Motion [revs per day]
        mean_motion = float(line2_split[7])
        # Revolution Number at Epoch
        # rev_num_at_epoch = int(line2_split[8])

    # Take out what is needed and display each of the properties
    print(f"Inclination angle (degrees):                             {inclination}")
    print(f"Right Ascension of Ascending Node (RAAN) (degrees):      {raan}")
    print(f"Eccentricity:                                            {eccentricity:.5f}")
    print(f"Mean anomaly (degrees):                                  {mean_anomaly:.3f} ")
    print(f"Mean motion (rev/day):                                   {mean_motion:.3f} ", end="\n\n")

    return (inclination, raan, eccentricity, arg_perigee,
            mean_anomaly, mean_motion)

def calculate_orbital_parameters(eccentricity : float, mean_motion : float, mu : float,
                                 earth_rad : float) -> tuple[float] :
    """Calculates more extensive orbital parameters of a satellite
    Args:
        eccentricity (float)
        mean_motion (float): Revolutions per day
        mu (float): Gravitational constant

    Returns:
        tuple[float]: period and semi-major axis
    """
    # Convert mean motion from revs/day to seconds/rev
    period = (1/mean_motion)* 24 * 3600

    # Calculate semi-major axis (a) using Kepler's third law
    semimajor_axis = (period * np.sqrt(mu)/(2 * np.pi)) ** (2/3)

    # Calculate perigee and apogee distances
    r_perigee = semimajor_axis * (1 - eccentricity)
    alt_perigee = r_perigee - earth_rad
    r_apogee = semimajor_axis * (1 + eccentricity)
    alt_apogee = r_apogee - earth_rad

    # Calculate angular momentum (h)
    angular_momentum = np.sqrt(semimajor_axis * mu * (1 - eccentricity**2))

    # Display each of the properties
    print(f"Perigee altitude (km):             {alt_perigee:.3f}")
    print(f"Apogee altitude (km):              {alt_apogee:.3f}")
    print(f"Semi-major axis (km):              {semimajor_axis:.3f} ")
    print(f"Orbital period (s):                {period:.3f} ")
    print(f"Angular momentum (km^2/s):         {angular_momentum:.3f}", end="\n\n")

    return period, semimajor_axis, r_perigee, r_apogee, angular_momentum

def check_intersection(orbit_1: tuple, orbit_2: tuple, i_return: int = 0,
                       tolerance: float = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to check for intersection points between two orbits.

    Args:
        orbit_1 (tuple): The first orbit, as a tuple of (x, y, z) arrays.
        orbit_2 (tuple): The second orbit, as a tuple of (x, y, z) arrays.
        tolerance (float): The tolerance within which two points are considered to intersect. Defaults to 10 units.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two intersection points (r_int1, r_int2) if found, otherwise returns arrays filled with zeros.
    """
    x1, y1, z1 = orbit_1
    x2, y2, z2 = orbit_2

    r_int1 = np.zeros(3)
    r_int2 = np.zeros(3)
    i1_orb1, i2_orb1 = 0, 0

    found_first_intersection = False

    # Loop through points in the first orbit
    for i in range(len(x1)):
        # Loop through points in the second orbit
        for j in range(len(x2)):
            # Calculate differences in x, y, z coordinates
            x_diff = x1[i] - x2[j]
            y_diff = y1[i] - y2[j]
            z_diff = z1[i] - z2[j]

            # Calculate the magnitude of the difference
            mag_difference = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

            # If the distance is less than the tolerance, we've found an intersection
            if mag_difference < tolerance:
                if not found_first_intersection:
                    r_int1[0], r_int1[1], r_int1[2] = x1[i], y1[i], z1[i]
                    found_first_intersection = True  # Mark the first intersection as found
                    i1_orb1 = i
                else:
                    r_int2[0], r_int2[1], r_int2[2] = x1[i], y1[i], z1[i]
                    i2_orb1 = i
                    if i_return:
                        return r_int1, r_int2, i1_orb1, i2_orb1
                    return r_int1, r_int2  # Return both intersections when found

    # If only one intersection is found, the second will remain (0, 0, 0)
    return r_int1, r_int2



def fix_orbit(orbit: tuple, r_start: np.ndarray, r_finish: np.ndarray,
              check: int, tolerance: float = 10) -> tuple:
    """
    Function to trim a satellite's orbit between two radius vectors r_start and r_finish.

    Args:
        orbit (tuple): The full satellite orbit in the form of (x, y, z) arrays.
        r_start (np.ndarray): Starting position (x, y, z) where the orbit should begin.
        r_finish (np.ndarray): Ending position (x, y, z) where the orbit should end.
        check (int): Flag for printing the magnitude difference (for debugging). Defaults to 0.
        tolerance (float): Tolerance for detecting start and finish points. Defaults to 30 units.

    Returns:
        tuple: The trimmed orbit (x_fix, y_fix, z_fix).
    """
    x, y, z = orbit
    reach_start = False
    i_start, i_finish = 0, 0

    x_fix, y_fix, z_fix = [],[],[]

    for i in range(len(x)):
        # Compute the difference vectors
        x_diff_start, y_diff_start, z_diff_start = x[i] - r_start[0], y[i] - r_start[1], z[i] - r_start[2]
        x_diff_finish, y_diff_finish, z_diff_finish = x[i] - r_finish[0], y[i] - r_finish[1], z[i] - r_finish[2]

        # Magnitude of the difference vectors
        mag_difference_start = np.sqrt(x_diff_start**2 + y_diff_start**2 + z_diff_start**2)
        mag_difference_finish = np.sqrt(x_diff_finish**2 + y_diff_finish**2 + z_diff_finish**2)

        if check == 1 and mag_difference_finish < 100:
            print(f"Start difference at index {i}: {mag_difference_finish}")

        # Check if we've reached the start point
        if mag_difference_start < tolerance and not reach_start:
            if check == 1 :
                print("hey baby")
                print(i)
                print(x[i], y[i], z[i])
            i_start = i
            reach_start = True
            continue

        # Check if we've reached the finish point
        if mag_difference_finish < tolerance and reach_start:
            if check == 1 :
                print("sad")
                print(i, reach_start)
                print(x[i], y[i], z[i])
            i_finish = i
            break

        # Append orbit points if within the start and before the finish
        if reach_start:
            x_fix.append(x[i])
            y_fix.append(y[i])
            z_fix.append(z[i])

    i_diff = abs(i_finish - i_start)
    # Convert lists back to numpy arrays for consistency
    return np.array(x_fix), np.array(y_fix), np.array(z_fix), i_diff


def delta_vs(chase, target, mu):
    (chase_name, r_chase_perigee, r_chase_apogee, period_chase,
     h_chase, inc_ang_chase, raan_chase) = chase
    (targ_name, r_targ_perigee, r_targ_apogee, period_targ,
     h_targ, inc_ang_targ, raan_targ) = target

    # All orbits are quite circular, assume that the radius of each orbit
    # Is the average of its apogee and perigee (semimajor axis)
    # This will give us an approximate 1km error
    semimajor_axis_chase = (r_chase_perigee+r_chase_apogee)/2

    h_mid_ellipse = form.angular_momentum(r_chase_perigee, r_targ_apogee, mu)

    delta_v_inc = form.delta_plane(h_chase, semimajor_axis_chase, inc_ang_chase, inc_ang_targ)
    delta_v_raan = form.delta_plane(h_chase, semimajor_axis_chase, raan_chase, raan_targ)
    delta_v_init = form.delta_v(h_chase, h_mid_ellipse, r_chase_perigee)
    delta_v_fin = form.delta_v(h_mid_ellipse, h_targ, r_targ_apogee)
    delta_v_hohmann = delta_v_init + delta_v_fin
    delta_v_total = delta_v_inc + delta_v_raan + delta_v_hohmann

    print("\033[4m" + "Transfer values:" + "\033[0m")
    print(f"Velocity change for Inclination change (km/s):     {(delta_v_inc):.3f}")
    print(f"Velocity change for RAAN change (km/s):            {(delta_v_raan):.3f}")
    print(F"Velocity change to enter Hohmann (km/s)            {(delta_v_init):.3f}")
    print(F"Velocity change to exit Hohmann (km/s)             {(delta_v_fin):.3f}")
    print(f"Velocity change for Hohmann (km/s):                {(delta_v_hohmann):.3f}")
    print(f"Total delta v required (km/s):                     {(delta_v_total):.3f}", end = '\n\n')

    return delta_v_total

def sort_orb_efficiency(orbit_org : tuple, orbits : list, omega_e : float,
                        points_sim : float, m0 : float, earth_rad : float, mu : float):
    # Unpack original orbit
    (orbit_name, r_org, r_org, period_org,
    h_org, inc_ang_org, raan_org) = orbit_org

    # Create array to store total delta v for each possible orbit transfer
    total_delta_v = np.zeros(len(orbits))

    for i in range(len(orbits)):
        v_total = delta_vs(orbit_org, orbits[i], mu)
        total_delta_v[i] = v_total

    v_min = np.min(total_delta_v)
    v_i_min = np.argmax(total_delta_v)

    print(f"Transferring to {orbits[v_i_min][0]} \n \n")

    i_diff, period_mid = orb_sim.sim_delta_time(current_orbit, orbits[v_i_min],
                                omega_e, points_sim, mu)

    period_current = current_orbit[5]
    total_time = form.total_time(period_current, period_mid, i_diff, points_sim, T_return = 1)
    phase_sim.phase_sim(total_time, orbits[v_i_min],  m0,  earth_rad, mu)

    # Remove orbit that has already been reached
    current_orbit = orbits[v_i_min]

    del orbits[v_i_min]

    return orbits, current_orbit, v_min
