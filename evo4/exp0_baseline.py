# =========================
## EXPERIMENT 0 (BASELINE)
# Task: Making the gecko walk as far as possible in the environment BoxyRugged.
# Fitness = XY displacement of the "core" body after a fixed number of steps.
# =========================

import numpy as np
import matplotlib.pyplot as plt
import os
from utils import rollout_fitness, moving_average, NUM_JOINTS
from params import GENERATIONS, SEEDS

# Derive number of runs directly from SEEDS
NUM_RUNS = len(SEEDS)

def run_one_seed(seed):
    """
    Running the baseline (random search) once with a fixed seed.

    """
    rng = np.random.default_rng(seed)
    best_per_gen = []
    best_overall = -np.inf

    for g in range(GENERATIONS):
        # Generate a random solution (baseline search) 
        # with param vector size: 2 * NUM_JOINTS + 1
        x = rng.normal(size=NUM_JOINTS * 2 + 1)
        f = rollout_fitness(x)

        # Update bests
        best_overall = max(best_overall, f)
        best_per_gen.append(best_overall)

        # Quick progress print
        print(f"[seed {seed}] gen {g+1:02d}/{GENERATIONS}  best={best_overall:.4f} m")

    return np.array(best_per_gen, dtype=float), float(best_overall)


def main():
    """Main: run NUM_RUNS seeds and plot mean±std"""
    results_dir = "__results__"
    os.makedirs(results_dir, exist_ok=True)

    curves = []   # Will become shape (NUM_RUNS, generations)
    bests  = []   # One best number per seed

    for seed in SEEDS:
        curve, best = run_one_seed(seed)
        curves.append(curve)
        bests.append(best)

    curves = np.vstack(curves)            # (NUM_RUNS, generations)
    mean = curves.mean(axis=0)            # Mean over seeds, per generation
    std  = curves.std(axis=0)             # Std over seeds, per generation

    # Quick report in terminal
    print(f"\nExperiment run with {NUM_RUNS} seeds")
    print("Best distances per seed:", [f"{b:.4f}" for b in bests])
    print(f"Final mean (gen {GENERATIONS}) = {mean[-1]:.4f} ± {std[-1]:.4f} m")

    # Plotting
    xs = np.arange(1, GENERATIONS + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(xs, mean, label="Baseline (mean of runs)")
    plt.fill_between(xs, mean - std, mean + std, alpha=0.25, label="±1 std")

    smoothed = moving_average(mean, w=5)
    xs_smooth = np.arange(1, len(smoothed) + 1)

    plt.plot(xs_smooth, smoothed, linewidth=2,
        label="Baseline (mean, moving avg w=5)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness = XY displacement (m)")
    plt.title("Experiment 0 — Baseline")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/exp0_baseline.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
