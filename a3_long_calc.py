"""
a2_long_calc.py

This files contains certain calculations that require more than 1-2 lines of working out,
or calculating multiple values at once
"""
from scipy import constants as spconst
from scipy import integrate
import numpy as np
import random
import os
from orbit_object import Orbit as orb_obj
import a3_formulae as form
import a3_orbit_sim as orb_sim
import a3_phase_sim as phase_sim


def random_tle(satellite_number):
    # Randomly generate orbital parameters
    inclination = random.uniform(0, 180)  # Inclination in degrees
    ra_of_asc_node = random.uniform(0, 360)  # Right Ascension of Ascending Node in degrees
    eccentricity = random.uniform(0, 0.1)  # Eccentricity (0 to 1)
    argument_of_perigee = random.uniform(0, 360)  # Argument of Perigee in degrees
    mean_anomaly = random.uniform(0, 360)  # Mean Anomaly in degrees
    mean_motion = random.uniform(13, 16)  # Mean motion (revolutions per day)

    # Format TLE according to the given format
    line1 = f'1 {satellite_number:05d}U 95035B {random.uniform(24000, 25000):.8f} {random.uniform(-0.00001, 0.00001):.8f} 00000+0 00000+0 0 9993'
    line2 = f'2 {satellite_number:05d} {inclination:8.4f} {ra_of_asc_node:8.4f} {eccentricity:07.7f} {argument_of_perigee:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f} 107176'

    return line1, line2

def write_random_tles_to_files(num_tles):
    # Create a directory to store TLE files if it doesn't exist
    os.makedirs('TLE_Files', exist_ok=True)

    for i in range(num_tles):
        satellite_number = random.randint(10000, 99999)  # Generate a random satellite number
        line1, line2 = random_tle(satellite_number)

        # Create a filename for the TLE
        filename = f'TLE_Files/TLE_{i}.txt'

        # Write the TLE to the file
        with open(filename, 'w') as f:
            f.write(f"{line1}\n{line2}\n")  # Write both lines to the file

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

        if file_name[0] == "O" or file_name[0] == "W":
            # Eccentricity (Assumed leading decimal)
            eccentricity = float("0." + line2_split[4])
        else:
            # Eccentricity (Assumed leading decimal)
            eccentricity = float(line2_split[4])
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

def check_intersection(orbit_1: tuple, orbit_2: tuple, print_true: int = 0,
                       tolerance: float = 50) -> tuple[np.ndarray, np.ndarray]:
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

            if mag_difference < 500 and print_true:
                print(mag_difference)
            # If the distance is less than the tolerance, we've found an intersection
            if mag_difference < tolerance:
                if not found_first_intersection:
                    r_int1[0], r_int1[1], r_int1[2] = x1[i], y1[i], z1[i]
                    found_first_intersection = True  # Mark the first intersection as found
                    i1_orb1 = i
                else:
                    r_int2[0], r_int2[1], r_int2[2] = x1[i], y1[i], z1[i]
                    i2_orb1 = i
                    # if i_return:
                    #     return r_int1, r_int2, i1_orb1, i2_orb1
                    if print_true:
                        print("Chase and target orbits intersect!!!")
                    return r_int1, r_int2  # Return both intersections when found

    # If only one intersection is found, the second will remain (0, 0, 0)
    return r_int1, r_int2



def fix_orbit(orbit: tuple, r_start: np.ndarray, r_finish: np.ndarray,
              check: int, tolerance: float = 50) -> tuple:
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

def delta_vs(current, target, m0, isp, mu):
    cur_orb = orb_obj(current, mu)
    targ_orb = orb_obj(target, mu)

    # Two for Rise orbit, One for combo, One for hohmann, One for phase
    delta_transfers = np.zeros(5)

    # All orbits are quite circular, assume that the radius of each orbit
    # Is the average of its apogee and perigee (semimajor axis)
    # This will give us an approximate 1km error

    h_mid_ellipse = form.angular_momentum(cur_orb.r_p, targ_orb.r_a, mu)
    h_rise = form.angular_momentum(targ_orb.r_a, targ_orb.r_a, mu)

    # Manuevres in order (no simulation, just theoritical delta-v  values)

    # Step 1: Raise the orbit to target orbit's apogee
    delta_transfers[0] = form.delta_v(cur_orb.h, h_mid_ellipse, cur_orb.r_p)
    delta_transfers[1] = form.delta_v(h_mid_ellipse, h_rise, targ_orb.r_p)

    # Step 2: Conduct combined plane (and hohmann) transfer at orbit apogee
    delta_transfers[2] = form.delta_comb_plane(h_rise, targ_orb.r_a, cur_orb.i, targ_orb.i,
                                        cur_orb.raan, targ_orb.raan, print_stuff = 1)
    delta_transfers[3] = form.delta_v(h_rise, targ_orb.h, targ_orb.r_a)

    print("\033[4m" + "Transfer values from " + cur_orb.name + " to " + targ_orb.name + ": \033[0m")
    print(F"Velocity change to enter rised orbit (km/s)        {(delta_transfers[0]):.3f}")
    print(F"Velocity change to exit rised orbit (km/s)         {(delta_transfers[1]):.3f}")
    print(f"Velocity change for plane combo change (km/s):     {(delta_transfers[2]):.3f}")
    print(f"Velocity change to lower orbit (km/s):             {(delta_transfers[3]):.3f}")

    return delta_transfers

def mission_total_v(chase, targ, points_sim, m0, isp, thrust,
                    earth_rad, omega_e, mu, park: int = 0):

    v_transfers = delta_vs(chase, targ, m0, isp, mu)

    i_diff, period_mid, period_rise = orb_sim.sim_delta_time(chase, targ,
                                omega_e, points_sim, mu)

    t_transfers = form.total_time(period_mid, period_rise, i_diff, points_sim)
    v_phase, t_phase = phase_sim.phase_sim(t_transfers, targ,  earth_rad, mu, print_v = 0)

    v_total = v_transfers + v_phase
    if park:
        v_transfers[4] = v_phase
        v_total = v_transfers
        t_total = t_transfers + t_phase
    else:
        v_total = v_transfers
        t_total = t_transfers

    print(f"Total delta v required (km/s):                     {(np.linalg.norm(v_total)):.3f}", end = '\n\n')

    return v_total, t_total

def fuel_per_transfer(delta_vs, m0, isp):
    delta_fuel_rise_init, fuel_left_rise_init = form.change_in_mass(delta_vs[0], m0, isp)
    delta_fuel_rise_fin, fuel_left_rise_fin = form.change_in_mass(delta_vs[1], fuel_left_rise_init, isp)
    delta_fuel_combo, fuel_left_combo = form.change_in_mass(delta_vs[2], fuel_left_rise_fin, isp)
    delta_fuel_lower, fuel_left_lower = form.change_in_mass(delta_vs[3], fuel_left_combo, isp)
    delta_fuel_phase, fuel_left_phase = form.change_in_mass(delta_vs[4], fuel_left_lower, isp)

    total_fuel = delta_fuel_rise_init + delta_fuel_rise_fin \
                + delta_fuel_combo + + delta_fuel_lower + delta_fuel_phase

    return total_fuel, fuel_left_phase

def sort_orb_efficiency(park_orbit : tuple, orbits : list, omega_e : float,
                        points_sim : float, m0 : float, isp : float, thrust : float,
                        earth_rad : float, mu : float):

    # Initialize arrays with the correct syntax
    # 4 columns for delta-v is to store each transfer
    park_delta_v = np.zeros((len(orbits), 5))     # 2D array for park delta-v values with 4 columns
    park_time = np.zeros(len(orbits))             # 1D array for park time
    park_fuel = np.zeros(len(orbits))             # 1D array for park fuel

    transfer_delta_v = np.zeros((len(orbits), len(orbits), 5))   # 3D array for transfer delta-v values
    transfer_time = np.zeros((len(orbits), len(orbits)))         # 2D array for transfer times
    transfer_fuel = np.zeros((len(orbits), len(orbits)))         # 2D array for transfer fuel

    total_delta_v = np.zeros((len(orbits), len(orbits), len(orbits)))   # 3D array for total delta-v
    total_fuel = np.zeros((len(orbits), len(orbits), len(orbits)))      # 3D array for total fuel

    for i in range(len(orbits)):
        # Finds delta_v to exit inital parking orbit
        delta_park, t_park = mission_total_v(park_orbit, orbits[i], points_sim, m0, isp,
                                    thrust, earth_rad, omega_e, mu, park = 1)
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
                                            thrust, earth_rad, omega_e, mu)
                transfer_delta_v[i][j] = delta_v2
                transfer_time[i][j] = t_transfer
            else:
                continue

    # Calculate every possible 'total delta-v' into a 2D array
    for i in range(len(orbits)):
        park_fuel[i], fuel_left_park = fuel_per_transfer(park_delta_v[i], m0, isp)
        for j in range(len(orbits)):
            # Phase First to Second orbit
            time_elapse1 = park_time[i] + transfer_time[i][j]
            v_phase1, t_phase1 = phase_sim.phase_sim(time_elapse1, orbits[j], earth_rad, mu)
            transfer_delta_v[i][j][4] = v_phase1    # Slot phase velocity into 4th slot

            # Now that we have phase, we must calculate the fuel required for our mission
            transfer_fuel[i][j], fuel_left_1 = \
                                fuel_per_transfer(transfer_delta_v[i][j], fuel_left_park, isp)

            for k in range(len(orbits)):
                if i != j and j !=k and i != k:
                    time_elapse2 = park_time[i] + transfer_time[i][j] + transfer_time[j][k]
                    v_phase2, t_phase2 = phase_sim.phase_sim(time_elapse2, orbits[k], earth_rad, mu)
                    transfer_delta_v[j][k][4] = v_phase2

                    transfer_fuel[j][k], fuel_left_2 = \
                                fuel_per_transfer(transfer_delta_v[j][k], fuel_left_1, isp)

                    total_delta_v[i][j][k] = np.linalg.norm(park_delta_v[i]) + np.linalg.norm(transfer_delta_v[i][j]) + v_phase1 \
                                            + np.linalg.norm(transfer_delta_v[j][k]) + v_phase2
                    total_fuel[i][j][k] = park_fuel[i] + transfer_fuel[i][j] + transfer_fuel[j][k]
                else:
                    # Use np.nan for easier filtering later
                    total_delta_v[i][j][k] = np.nan

    min_index = np.unravel_index(np.nanargmin(total_delta_v), total_delta_v.shape)
    min_value = total_delta_v[min_index]
    min_fuel = total_fuel[min_index]

    print(f"Transferring to                                     {orbits[min_index[0]][0]}")
    print(f"Then transferring to                                {orbits[min_index[1]][0]}")
    print(f"Then transferring to                                {orbits[min_index[2]][0]}")

    print(f"Phasing for first transfer (km/s):                  {park_delta_v[min_index[0]][4]:.3f}")
    print(f"Phasing for second transfer (km/s):                 {transfer_delta_v[min_index[:2]][4]:.3f}")
    print(f"Phasing for third transfer (km/s):                  {transfer_delta_v[min_index[1:]][4]:.3f}")

    print(f"Fuel for first transfer (kg):                       {park_fuel[min_index[0]]:.3f}")
    print(f"Fuel for second transfer (kg):                      {transfer_fuel[min_index[:2]]:.3f}")
    print(f"Fuel for third transfer (kg):                       {transfer_fuel[min_index[1:]]:.3f}")

    print(f"Least fuel required (Isp = {isp}) (kg):             {min_fuel:.3f}")
    print(f"Total most efficient mission delta-v (km/s):        {min_value:.3f}")
