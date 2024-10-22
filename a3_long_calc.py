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
    print(f"Argument of Perigee                                      {arg_perigee}")
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

def apse_line(current : tuple, target : tuple, mu : float) -> float:
    cur_orb = orb_obj(current, mu)
    targ_orb = orb_obj(target, mu)

    # Find true anomaly of intersection
    delta_arg_per = abs(cur_orb.omega - targ_orb.omega)

    a = cur_orb.e * targ_orb.h**2 - targ_orb.e * cur_orb.h**2 * np.cos(delta_arg_per)
    b = -targ_orb.e * cur_orb.h**2 * np.sin(delta_arg_per)
    c = cur_orb.h**2 - targ_orb.h**2

    phi = np.arctan2(b, a)
    ang = (c / a) * np.cos(phi)

    if abs(ang) > 180:
        ang = ang - 360

    theta_1 = phi + np.arccos(np.clip(np.radians(ang), -1, 1))  # Use np.clip to avoid acos domain errors

    # Radius at apse line intersection
    r = (cur_orb.h**2 / mu) * (1 / (1 + cur_orb.e * np.cos(theta_1)))

    # Velocity components and flight path of orbit 1 (current orbit)
    v_perp1 = cur_orb.h / r
    v_r1 = (mu / cur_orb.h) * cur_orb.e * np.sin(theta_1)
    v_tot1 = np.sqrt(v_r1**2 + v_perp1**2)
    flight_ang_1 = np.arctan2(v_r1, v_perp1)

    # Velocity components and flight path of orbit 2 (target orbit)
    v_perp2 = targ_orb.h / r
    v_r2 = (mu / targ_orb.h) * targ_orb.e * np.sin(theta_1)
    v_tot2 = np.sqrt(v_r2**2 + v_perp2**2)
    flight_ang_2 = np.arctan2(v_r2, v_perp2)

    delta_v = np.sqrt(v_tot1**2 + v_tot2**2 - 2 * v_tot1 * v_tot2 * np.cos(flight_ang_2 - flight_ang_1))

    return delta_v

def delta_vs(current, target, m0, isp, mu):
    cur_orb = orb_obj(current, mu)
    targ_orb = orb_obj(target, mu)

    # All orbits are quite circular, assume that the radius of each orbit
    # Is the average of its apogee and perigee (semimajor axis)
    # This will give us an approximate 1km error
    semimajor_axis_chase = (targ_orb.r_p + targ_orb.r_a)/2

    h_mid_ellipse = form.angular_momentum(cur_orb.r_p, targ_orb.r_a, mu)

    delta_v_inc = form.delta_plane(cur_orb.h, semimajor_axis_chase, cur_orb.i, targ_orb.i)
    apse_vs = apse_line(current, target, mu)
    delta_v_init = form.delta_v(cur_orb.h, h_mid_ellipse, cur_orb.r_p)
    delta_v_fin = form.delta_v(h_mid_ellipse, targ_orb.h, targ_orb.r_a)
    delta_v_hohmann = delta_v_init + delta_v_fin
    delta_v_raan = form.delta_plane(cur_orb.h, targ_orb.r_a, cur_orb.raan, targ_orb.raan, print_stuff = 1)
    delta_v_total = delta_v_inc + delta_v_raan + delta_v_hohmann + apse_vs

    print("\033[4m" + "Transfer values from " + cur_orb.name + " to " + targ_orb.name + ": \033[0m")
    print(f"Velocity change for Inclination change (km/s):     {(delta_v_inc):.3f}")
    print(f"Velocity change for Apse Line Rotation (km/s):     {apse_vs:.3f}")
    print(F"Velocity change to enter Hohmann (km/s)            {(delta_v_init):.3f}")
    print(F"Velocity change to exit Hohmann (km/s)             {(delta_v_fin):.3f}")
    print(f"Velocity change for Hohmann (km/s):                {(delta_v_hohmann):.3f}")
    print(f"Velocity change for RAAN change (km/s):            {(delta_v_raan):.3f}")

    return delta_v_total

def mission_total_v(chase, targ, points_sim, m0, isp, earth_rad, omega_e, mu, park: int = 0):
    chase_orb = orb_obj(chase, mu)

    v_transfers = delta_vs(chase, targ, m0, isp, mu)

    i_diff, period_mid = orb_sim.sim_delta_time(chase, targ,
                                omega_e, points_sim, mu)

    period_current = chase_orb.T
    t_transfers = form.total_time(period_current, period_mid, i_diff, points_sim, T_return = 1)
    v_phase, t_phase = phase_sim.phase_sim(t_transfers, targ,  m0,  earth_rad, mu, print_v = 0)

    v_total = v_transfers + v_phase
    if park:
        v_total = v_transfers + v_phase
        t_total = t_transfers + t_phase
    else:
        v_total = v_transfers
        t_total = t_transfers
    print(f"Total delta v required (km/s):                     {(v_total):.3f}", end = '\n\n')

    return v_total, t_total

def sort_orb_efficiency(park_orbit : tuple, orbits : list, omega_e : float,
                        points_sim : float, m0 : float, isp : float, earth_rad : float, mu : float):

    # Create array to store total delta v for each possible orbit transfer
    # Extra row to take into account parking orbits
    park_delta_v =  np.zeros(len(orbits))
    park_time = np.zeros(len(orbits))

    transfer_delta_v = np.zeros((len(orbits), len(orbits)))
    transfer_time = np.zeros((len(orbits), len(orbits)))

    total_delta_v = np.zeros((len(orbits), len(orbits), len(orbits)))

    for i in range(len(orbits)):
        # Finds delta_v to exit inital parking orbit
        delta_park, t_park = mission_total_v(park_orbit, orbits[i], points_sim, m0, isp,
                                    earth_rad, omega_e, mu, park = 1)
        park_delta_v[i] = delta_park
        park_time[i] = t_park

        for j in range(len(orbits)):
            # We do not want to calculate the delta-v of the same orbit and we also don't
            # want to repeat calculations
            if i != j:
                # Keeping track of index
                #print(i,j)
                # Calculate delta-v between two unique orbits
                delta_v2, t_transfer = mission_total_v(orbits[i], orbits[j], points_sim, m0, isp,
                                            earth_rad, omega_e, mu)
                transfer_delta_v[i][j] = delta_v2
                transfer_time[i][j] = t_transfer
            else:
                continue

    # Calculate every possible 'total delta-v' into a 2D array
    for i in range(len(orbits)):
        for j in range(len(orbits)):
            # Phase First to Second orbit
            time_elapse1 = park_time[i] + transfer_time[i][j]
            v_phase1 = phase_sim.phase_sim(time_elapse1, orbits[j], m0, earth_rad, mu)
            for k in range(len(orbits)):
                if i != j and j !=k and i != k:
                    time_elapse2 = park_time[i] + transfer_time[i][j] + transfer_time[j][k]
                    v_phase2 = phase_sim.phase_sim(time_elapse2, orbits[k], m0, earth_rad, mu)
                    total_delta_v[i][j][k] = park_delta_v[i] + transfer_delta_v[i][j] + transfer_delta_v[j][k] \
                                                + v_phase1[0] + v_phase2[0]
                else:
                    # Use np.nan for easier filtering later
                    total_delta_v[i][j][k] = np.nan

    min_index = np.unravel_index(np.nanargmin(total_delta_v), total_delta_v.shape)
    min_value = total_delta_v[min_index]
    print(total_delta_v)

    # Dynamically print the transfer process based on the index
    print(f"Transferring to         {orbits[min_index[0]][0]}")
    print(f"Then transferring to    {orbits[min_index[1]][0]}")
    print(f"Then transferring to    {orbits[min_index[2]][0]}")
    return min_value
