import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.integrate import solve_ivp
from orbit_object import Orbit as orb_obj
import a3_formulae as form

def time_from_perigee(period : float, t_elapse : float):
    t_diff = t_elapse % period
    return t_diff

def period_phase(period_target : float, t_to_perigee : float, earth_rad :float, mu : float):
    # Threshold for period, 100km above earth's surface
    # since anything below this radius cannot be a
    # functioning orbit
    period_earth = form.period(earth_rad + 100, mu)
    period_phase = t_to_perigee
    if period_phase >= period_earth:
        return period_phase
    else:
        return period_phase + period_target

def apogee_rad(semimajor_axis : float, perigee : float):
    apogee = 2*semimajor_axis - perigee
    return apogee

# Define a function to calculate the orbit
def plot_phase_orbit(fig : Figure, ax : Axes, perigee : float, apogee : float,
                     label : str, earth_rad : float, show: int = 0):

    # Semi-major axis
    a = (perigee + apogee) / 2
    # Eccentricity
    e = (apogee - perigee) / (apogee + perigee)

    # Theta for polar coordinates
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Orbital radius for each angle
    r = a * (1 - e**2) / (1 + e * np.cos(theta))

    # Convert polar to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Plot the orbit
    ax.plot(x, y, label=f'Orbit: {label}')

    if show:
        # Create a filled circle
        circle = plt.Circle((0, 0), earth_rad, color='blue', alpha=0.4, label = "Earth")  # (x, y) position, radius, color, and transparency

        # Add the circle to the axes
        ax.add_artist(circle)
        # Formatting the plot
        plt.title('Visualising phasing maneuver of chaser satellite')
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.legend(loc='upper right')
        plt.grid(True)

        # Show the plot
        plt.show()

def phase_sim(time_elapsed_init : float, target : tuple, m0 : float,
              earth_rad : float, mu : float) -> float:
    """Simulates phasing maneuver

    Args:
        time_elapsed_init (float): Elapsed time since
        target (tuple): tuple that packs important parameters for target satellite
        r_targ_perigee (float): radisu of target satellite perigee
        r_targ_apogee (float): radius of target satellite apogee
        m0 (float): mass of satellite (wet)
        earth_rad (float): radius of Earth
        mu (float): Gravitational constant
    """
    targ_orb = orb_obj(target, mu)

    time_diff_init = time_from_perigee(targ_orb.T, time_elapsed_init)
    time_to_perigee = targ_orb.T - time_diff_init
    T_phase = period_phase(targ_orb.T, time_to_perigee, earth_rad, mu)
    semimajor_phase = form.semimajor_reversed(T_phase, mu)
    apogee_phase = apogee_rad(semimajor_phase, targ_orb.r_a)

    # print(f"Target orbit period(s)            {targ_orb.T:.3f}")
    # print(f"Target orbit apogee (km)          {targ_orb.r_a:.3f}")
    # print(f"Phase orbit period (s):           {T_phase:.3f}")
    # print(f"Phase orbit perigee (km):         {targ_orb.r_p:.3f}")
    # print(f"Phase orbit apogee (km):          {apogee_phase:.3f}")

    h_targ = form.angular_momentum(targ_orb.r_p, targ_orb.r_a, mu)
    h_phase = form.angular_momentum(targ_orb.r_p, apogee_phase, mu)

    delta_v_phase_targ = form.delta_v(h_targ, h_phase, targ_orb.r_p)
    #delta_v_phase_circ = form.delta_v(h_circ, h_phase, r_targ_perigee)

    # print(f"Delta v to enter phase:                              {delta_v_phase_targ:.3f}")
    # print(f"Delta v to exit phase:                               {delta_v_phase_targ:.3f}")
    print(f"Total delta v to phase from raised orbit (km/s):   {(delta_v_phase_targ*2):.3f}")

    return delta_v_phase_targ*2
