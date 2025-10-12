# --- Core EA hyperparameters (longer & stronger run) ---
GENERATIONS = 60          # enough for convergence (≥ 50 recommended in feedback)
POP_SIZE    = 64          # larger search space, slower but better
SEEDS       = [42]        # fixed for reproducibility

# --- CMA-ES hyperparameters ---
SIGMA_INIT  = 0.5         # not too wild, not too small

# --- Tracking (W&B) ---
ENTITY = "evo4"
PROJECT = "assignment3"
CONFIG = {
    "Generations": GENERATIONS,
    "Population Size": POP_SIZE,
    "Initial Sigma": SIGMA_INIT,
}
