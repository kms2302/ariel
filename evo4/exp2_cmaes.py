# =========================
# EXPERIMENT 2 (CMA-ES)
# Task: Making the gecko walk as far as possible in the environment BoxyRugged.
# Fitness = XY displacement of the "core" body after a fixed number of steps.
# =========================

import math
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import cma
import os

# Same directory
from utils import NUM_JOINTS, rollout_fitness, moving_average, init_param_vec
from params import GENERATIONS, POP_SIZE, SEEDS, SIGMA_INIT


def run_one_seed(seed):
    """
    Running CMA-ES once (with a fixed random seed) and return:
      - Best_fitness_per_generation: list of floats, length = GENERATIONS
      - Best_overall: one float
    """
    rng = np.random.default_rng(seed)
    x0 = init_param_vec(rng)

    # Setting up CMA-ES
    # Note: CMA-ES MINIMIZES by default, so we'll pass negative fitness values
    es = cma.CMAEvolutionStrategy(
        x0,
        SIGMA_INIT,
        {'popsize': POP_SIZE, 'seed': seed, 'verbose': -9}  # quiet logs
    )

    best_per_gen = []

    for g in range(GENERATIONS):
        # ask(): sampling a whole population of candidate solutions
        population = es.ask()

        # Evaluating everyone; we negate because CMA-ES expects a "loss"
        losses = []
        for x in population:
            f = rollout_fitness(np.asarray(x, dtype=np.float64))  # Displacement (bigger is better)
            losses.append(-f)  # Negate. Lower is better for CMA

        # tell(): giving CMA-ES the evaluated losses so it can update its search distribution
        es.tell(population, losses)

        # For plotting: keeping the best (i.e., max displacement) this generation
        gen_best = -min(losses)  # Undo the minus sign
        best_per_gen.append(gen_best)

        # Quick progress print
        print(f"[seed {seed}] gen {g+1:02d}/{GENERATIONS}  best={gen_best:.4f} m")

    # CMA tracks the global best as `es.best.f` (this is the min loss), so flip sign again
    best_overall = -es.best.f
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

    np.save(f"{results_dir}/exp2_cmaes_mean.npy", mean)
    np.save(f"{results_dir}/exp2_cmaes_std.npy", std)

    # Quick report in terminal
    print("\nBest distances per seed:", [f"{b:.4f}" for b in bests])
    print(f"Final mean (gen {GENERATIONS}) = {mean[-1]:.4f} ± {std[-1]:.4f} m")

    # Plotting the figure: mean with a shaded ±1 std band
    xs = np.arange(1, GENERATIONS + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(xs, mean, label="CMA-ES (mean of 3 runs)")
    plt.fill_between(xs, mean - std, mean + std, alpha=0.25, label="±1 std")

    smoothed = moving_average(mean, w=5)
    xs_smooth = np.arange(1, len(smoothed) + 1)

    plt.plot(xs_smooth, smoothed, linewidth=2,
        label="CMA-ES (mean, moving avg w=5)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness = XY displacement (m)")
    plt.title("Experiment 2 — CMA-ES on Gecko (BoxyRugged)")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/exp2_cmaes_student.png", dpi=160)  # Save one clean figure
    plt.show()


if __name__ == "__main__":
    main()
