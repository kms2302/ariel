import math

# These are the basic hyperparams
TIMESTEPS   = 2000      # This is how many physics steps per rollout (longer = more time to move)
GENERATIONS = 30        # This is how many CMA-ES generations
POP_SIZE    = 32        # These are the candidates per generation
SEEDS       = [42, 1337, 2025]  # We run the whole experiment 3 times (for mean±std)
SIGMA_INIT  = 0.7       # The initial CMA-ES step size (how "wide" the search starts)
AMP_MAX = math.pi / 2   # Allowing up to 90 degrees per joint (big enough to push ground)
