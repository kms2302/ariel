# The basic hyperparameters
GENERATIONS = 4        # This is how many generations for the EA
POP_SIZE    = 256        # These are the candidates per generation
ALGOS       = ["CMA-ES"]
SEEDS       = [42, 1337, 2025]  # We run each whole experiment 3 times (for mean±std)

# The Weights & Biases parameters
ENTITY = "evo4"
PROJECT = "assignment3"
CONFIG = {
    "Generations": GENERATIONS,
    "Population Size": POP_SIZE,
}

# The CMA-ES hyperparameters
SIGMA_INIT  = 0.7       # The initial CMA-ES step size (how "wide" the search starts)
