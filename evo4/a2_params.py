# The basic hyperparameters
TIMESTEPS   = 7500      # This is how many physics steps per rollout (longer = more time to move)
GENERATIONS = 30        # This is how many generations for the EA
POP_SIZE    = 32        # These are the candidates per generation
ALGOS       = ["Baseline", "GA", "CMA-ES"]
SEEDS       = [42, 1337, 2025]  # We run each whole experiment 3 times (for mean±std)

# The Weights & Biases parameters
ENTITY = "evo4"
PROJECT = "assignment2"
CONFIG = {
    "Timesteps": TIMESTEPS,
    "Generations": GENERATIONS,
    "Population Size": POP_SIZE,
}

# The Genetic Algorithm hyperparameters
MUT_RATE    = 0.1   # per-gene mutation rate
MUT_STDEV   = 0.1   # mutation step size

# The CMA-ES hyperparameters
SIGMA_INIT  = 0.7       # The initial CMA-ES step size (how "wide" the search starts)
