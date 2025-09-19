# =========================
# EXPERIMENT 1 (GeneticAlgorithm)
# Task: Making the gecko walk as far as possible in the environment BoxyRugged.
# Fitness = XY displacement of the "core" body after a fixed number of steps.
# =========================

import math
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import os

# Same directory
from utils import NUM_JOINTS, rollout_fitness, moving_average, init_param_vec
from params import GENERATIONS, POP_SIZE, SEEDS, SIGMA_INIT


def run_one_seed(seed):
    """
    Running GeneticAlgorithm once (with a fixed random seed) and return:
      - Best_fitness_per_generation: list of floats, length = GENERATIONS
      - Best_overall: one float
    """
    x0 = init_param_vec(seed)

    # TODO: Setting up the GeneticAlgorithm
    # ...

    best_per_gen = []

    for g in range(GENERATIONS):
        # population = ...

        # Evaluating everyone; Don't negate f if GA doesn't expect a "loss"
        losses = []
        for x in population:
            f = rollout_fitness(np.asarray(x, dtype=np.float64))  # Displacement (bigger is better)
            losses.append(-f)  # Negated f

        # For plotting: keeping the best (i.e., max displacement) this generation
        gen_best = -min(losses)  # Undo the minus sign
        best_per_gen.append(gen_best)

        # Quick progress print
        print(f"[seed {seed}] gen {g+1:02d}/{GENERATIONS}  best={gen_best:.4f} m")

    # best_overall = ...
    return np.array(best_per_gen, dtype=float), float(best_overall)


def main():
    """Main: run 3 seeds and plot mean±std"""
    results_dir = "__results__"
    os.makedirs(results_dir, exist_ok=True)

    curves = []   # Will become shape (3, generations)
    bests  = []   # One best number per seed

    for seed in SEEDS:
        curve, best = run_one_seed(seed)
        curves.append(curve)
        bests.append(best)

    curves = np.vstack(curves)            # (n_seeds, generations)
    mean = curves.mean(axis=0)            # Mean over seeds, per generation
    std  = curves.std(axis=0)             # Std over seeds, per generation

    # Quick report in terminal
    print("\nBest distances per seed:", [f"{b:.4f}" for b in bests])
    print(f"Final mean (gen {GENERATIONS}) = {mean[-1]:.4f} ± {std[-1]:.4f} m")

    # Plotting the figure: mean with a shaded ±1 std band
    xs = np.arange(1, GENERATIONS + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(xs, mean, label="GA (mean of 3 runs)")
    plt.fill_between(xs, mean - std, mean + std, alpha=0.25, label="±1 std")

    smoothed = moving_average(mean, w=5)
    xs_smooth = np.arange(1, len(smoothed) + 1)
    
    plt.plot(xs_smooth, smoothed, linewidth=2,
        label="GA (mean, moving avg w=5)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness = XY displacement (m)")
    plt.title("Experiment 1 — GA on Gecko (BoxyRugged)")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/exp1_ga_student.png", dpi=160)  # Save one clean figure
    plt.show()


if __name__ == "__main__":
    main()

