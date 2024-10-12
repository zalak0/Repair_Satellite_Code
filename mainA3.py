import numpy as np
from itertools import permutations
from random_targets import random_targets
from transfer import transfer


def main() -> None:
    n = 1
    targets = random_targets(n)
    orders = list(permutations(range(1,n+1)))   # All possible orders of transferring orbits
    solutions = []
    for order in orders:                        # Iterative function for testing delta_v of each order
        order = [0, *order]                     # Ensure each attempt starts at LEO parking orbit
        time = 0                                # Time (s) 
        delta_v = 0                             # Delta_v (km/s)
        for o in range(len(order) - 1):                 # Iterative function to transfer between orbit pairs
            dv, time = transfer(targets[order[o]], targets[order[o+1]], time)  # Transfer function
            delta_v = delta_v + dv                                      
        solutions.append([order, delta_v, time])
    print(solutions)
    return

if __name__ == '__main__':
    main()


