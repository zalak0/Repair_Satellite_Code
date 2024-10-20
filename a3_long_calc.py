"""
a2_long_calc.py

This files contains certain calculations that require more than 1-2 lines of working out,
or calculating multiple values at once
"""
from scipy import constants as spconst
from scipy import integrate
import numpy as np
from orbit_object import Orbit as orb_obj
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


def delta_vs(current_orbit, target, m0, isp, mu):
    cur_orb = orb_obj(current_orbit, mu)
    targ_orb = orb_obj(target, mu)

    # All orbits are quite circular, assume that the radius of each orbit
    # Is the average of its apogee and perigee (semimajor axis)
    # This will give us an approximate 1km error
    semimajor_axis_chase = (cur_orb.r_p + cur_orb.r_a)/2

    h_mid_ellipse = form.angular_momentum(cur_orb.r_p, targ_orb.r_a, mu)

    delta_v_inc = form.delta_plane(cur_orb.h, semimajor_axis_chase, cur_orb.i, targ_orb.i)
    delta_v_raan = form.delta_plane(cur_orb.h, semimajor_axis_chase, cur_orb.raan, targ_orb.raan)
    delta_v_init = form.delta_v(cur_orb.h, h_mid_ellipse, cur_orb.r_p)
    delta_v_fin = form.delta_v(h_mid_ellipse, targ_orb.h, targ_orb.r_a)
    delta_v_hohmann = delta_v_init + delta_v_fin
    delta_v_total = delta_v_inc + delta_v_raan + delta_v_hohmann

    print("\033[4m" + "Transfer values from " + cur_orb.name + " to " + targ_orb.name + ": \033[0m")
    print(f"Velocity change for Inclination change (km/s):     {(delta_v_inc):.3f}")
    print(f"Velocity change for RAAN change (km/s):            {(delta_v_raan):.3f}")
    print(F"Velocity change to enter Hohmann (km/s)            {(delta_v_init):.3f}")
    print(F"Velocity change to exit Hohmann (km/s)             {(delta_v_fin):.3f}")
    print(f"Velocity change for Hohmann (km/s):                {(delta_v_hohmann):.3f}")

    return delta_v_total

def mission_total_v(chase, targ, points_sim, m0, isp, earth_rad, omega_e, mu, print_stuff: int = 1):
    chase_orb = orb_obj(chase, mu)

    v_transfers = delta_vs(chase, targ, m0, isp, mu)

    i_diff, period_mid = orb_sim.sim_delta_time(chase, targ,
                                omega_e, points_sim, mu)

    period_current = chase_orb.T
    transfer_time = form.total_time(period_current, period_mid, i_diff, points_sim, T_return = 1)
    phase_vs = phase_sim.phase_sim(transfer_time, targ,  m0,  earth_rad, mu)

    v_total = v_transfers + phase_vs
    print(f"Total delta v required (km/s):                     {(v_total):.3f}", end = '\n\n')

    return v_total

def sort_orb_efficiency(park_orbit : tuple, orbits : list, omega_e : float,
                        points_sim : float, m0 : float, isp : float, earth_rad : float, mu : float):

    # Create array to store total delta v for each possible orbit transfer
    # Extra row to take into account parking orbits
    park_delta_v =  np.zeros(len(orbits))
    transfer_delta_v = np.zeros(len(orbits))
    total_delta_v = np.zeros((len(orbits), len(orbits)))

    for i in range(len(orbits)):
        # Finds delta_v to exit inital parking orbit
        delta_park = mission_total_v(park_orbit, orbits[i], points_sim, m0, isp,
                                    earth_rad, omega_e, mu)
        park_delta_v[i] = delta_park

        for j in range(len(orbits)):
            # We do not want to calculate the delta-v of the same orbit and we also don't
            # want to repeat calculations
            if i != j and i < j and i != 0:
                # Keeping track of index
                print(i,j)
                # Calculate delta-v between two unique orbits
                delta_v2 = mission_total_v(orbits[i], orbits[j], points_sim, m0, isp,
                                            earth_rad, omega_e, mu, print_stuff = 0)
                transfer_delta_v[j] = delta_v2

            # Initial case, acts a bit different due to parking orbit
            elif i != j and i ==0:
                print(i,j)
                delta_v2 = mission_total_v(orbits[i], orbits[j], points_sim, m0, isp,
                                            earth_rad, omega_e, mu, print_stuff = 0)
                transfer_delta_v[j - 1] = delta_v2
            else:
                continue

    # Now that all the transfer values are extracted, the following calculates
    # every possible 'total delta-v' into a 2 dimensional array
    for i in range(len(orbits)):
        for j in range(len(orbits)):
            if i != j-1 and i != j:
                total_delta_v[i][j]= park_delta_v[i] + transfer_delta_v[j-1] + transfer_delta_v[j]
                # if i == 2 and j == 1:
                #     print(transfer_delta_v[j])
                #     print(transfer_delta_v[j-1])
                #     print(park_delta_v[i])
            else:
                # Placeholder, can be aslo replaced with nan
                total_delta_v[i][j] = 0

    # Create a mask for non-zero values
    non_zero_mask = total_delta_v > 0

    # Use the mask to filter non-zero values and find the minimum value
    min_non_zero_value = np.min(total_delta_v[non_zero_mask])

    # Get the index of the minimum non-zero value
    min_index = np.argwhere(total_delta_v == min_non_zero_value)
    min_value = total_delta_v[min_index[0][0], min_index[0][1]]

    print(total_delta_v)
    print(min_index)

    print(f"Transferring to         {orbits[min_index[0][0]][0]}")
    print(f"Then transferring to    {orbits[min_index[0][1] - 1][0]}")
    print(f"Then transferring to    {orbits[min_index[0][1]][0]}")

    return min_value
