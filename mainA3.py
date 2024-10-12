from itertools import permutations
from generate_random_targets import random_targets
from transfer import transfer


def main() -> None:
    """
    Generates a list of n satellite orbits, finding the delta v optimised
    path for visiting all satellites. 
    """
    # --- Generate Target Orbits ---
    n = 1                                       # Number of target orbits needed to visit
    targets = random_targets(n)                 # Generates a list of n target orbits
    orders = list(permutations(range(1,n+1)))   # All possible orders of transferring orbits listed by their index in the targets list

    # --- Test All Orbit Orders to Find Optimal Path ---
    solutions = []                              # A list of all solutions in format [path, delta_v_required, time_required]
    for order in orders:                        # Iterative function for testing delta_v of each order
        order = [0, *order]                     # Append LEO parking orbit to the front of the order
        total_time = 0                          # Time (s) 
        total_delta_v = 0                       # Delta_v (km/s)

        # Iterative function to transfer between orbit pairs
        for o in range(len(order) - 1):        
            # Function that finds delta_v and time required for each transfer pair
            delta_v, time = transfer(targets[order[o]], targets[order[o+1]], time)
            total_delta_v = total_delta_v + delta_v             # Keep track of total delta v required  
        solutions.append([order, delta_v, time])   # Append the path, delta v required and time required for this particular path
    print(solutions)    # Print all solutions
    return

if __name__ == '__main__':
    main()


