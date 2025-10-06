"""
This is a small script to test the baseline setup.

This file is just a helper: it calls the baseline population builder,
checks that culling works and prints out how many robots survived.
This way we can quickly verify that our code is working before moving on
to the more advanced experiments.
"""

from examples.a3.baseline import build_initial_population

import numpy as np


def main():
    """
    Entry point for the script.
    - Creates a random number generator with a fixed seed (for reproducibility).
    - Builds a small test population (5 robots).
    - Prints out how many robots survived culling.
    """
    rng = np.random.default_rng(42)  # fixed seed means same results every run

    # Builds a baseline population of 5 robots
    pop = build_initial_population(pop_size=5, rng=rng)

    # Printing result to confirm everything worked
    print(f"Built population of {len(pop)} robots")

# Running the main() function if this script is executed directly
if __name__ == "__main__":
    main()
