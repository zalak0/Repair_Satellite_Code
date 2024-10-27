"""
a2.py

This is the main file where all the code is run, the inputs
for each question are asked here and print_calc and plots questions
are run as a result.
"""
import numpy as np
import matplotlib.pyplot as plt
import a3_formulae as form
import a3_long_calc as lc

def main() -> None:
    """Main file
    """
    mu = 398600                         # Earth's gravitational parameter [km^3/s^2]
    earth_rad = 6371                    # Earth's mean radius [km]
    omega_e = 7.2921159 * 10**(-5)      # Earth's rotation

    # Amount of points being plotted in a given time frame
    # Here in this case, one period of a certain orbit
    # Need to simulate orbits to figure out exact location of satellite for phasing
    points_sim = 1500

    isp = 228.1
    thrust = 39 * 10**(-3)

    # The following is information for Questions 2, 3 and 4
    # Original satellite: Orbital parameters
    r_org = 305 + earth_rad                            # Circular orbit
    inc_ang_org, raan_org = 33.42, 144                     # Inclination angle and Right Angle of Ascending Node of original orbit
    period_org = form.period(r_org, mu)                 # Period of original orbit
    h_org = np.sqrt(r_org * mu)                         # Angular momentum of original orbit
    m0 = 1000                                             # Satellite wet mass (kg)

    lc.write_random_tles_to_files(3)

    # Extract Orbital parameters of each target satellite
    print("\033[4m" + "Orbit 1:" + "\033[0m")
    inc_ang_1, raan_1, eccentricity_1, arg_perigee_1, \
        mean_anomaly_1, mean_motion_1 = lc.deduce_tle("Orbit_TLEs/HST.txt")
    period_1, semimajor_axis_1, r_perigee_1, r_apogee_1, h_1 = \
        lc.calculate_orbital_parameters(eccentricity_1, mean_motion_1, mu, earth_rad)

    print("\033[4m" + "Orbit 2:" + "\033[0m")
    inc_ang_2, raan_2, eccentricity_2, arg_perigee_2, \
        mean_anomaly_2, mean_motion_2 = lc.deduce_tle("Orbit_TLEs/SORCE.txt")
    period_2, semimajor_axis_2, r_perigee_2, r_apogee_2, h_2 = \
        lc.calculate_orbital_parameters(eccentricity_2, mean_motion_2, mu, earth_rad)

    print("\033[4m" + "Orbit 3:" + "\033[0m")
    inc_ang_3, raan_3, eccentricity_3, arg_perigee_3, \
        mean_anomaly_3, mean_motion_3 = lc.deduce_tle("Orbit_TLEs/Terra.txt")
    period_3, semimajor_axis_3, r_perigee_3, r_apogee_3, h_3 = \
        lc.calculate_orbital_parameters(eccentricity_3, mean_motion_3, mu, earth_rad)

    # Pack each orbit for calculations
    orbit_org = ("Parking orbit", r_org, r_org, period_org,
                 h_org, inc_ang_org, raan_org, arg_perigee_1)
    orbit_1 = ("Orbit 1", r_perigee_1, r_apogee_1, period_1,
                h_1, inc_ang_1, raan_1, arg_perigee_1)
    orbit_2 = ("Orbit 2", r_perigee_2, r_apogee_2, period_2,
                h_2, inc_ang_2, raan_2, arg_perigee_2)
    orbit_3 = ("Orbit 3", r_perigee_3, r_apogee_3, period_3,
                h_3, inc_ang_3, raan_3, arg_perigee_3)

    # Pack the target orbits
    orbits = [orbit_1, orbit_2, orbit_3]

    # Set the current orbit to be the selected parking orbit
    lc.sort_orb_efficiency(orbit_org, orbits, omega_e, points_sim,
                           m0, isp, thrust, earth_rad, mu)

if __name__ == '__main__':
    main()
