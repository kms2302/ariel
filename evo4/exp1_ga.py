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
from utils import (
    NUM_JOINTS, 
    rollout_fitness, 
    moving_average, 
    init_param_vec,
    init_pop_vec,
)
from params import GENERATIONS, POP_SIZE, SEEDS, MUT_RATE, MUT_STDEV


def tournament_selection(pop, fitness, rng, k=3):
    """Pick one parent using k-way tournament."""
    idxs = rng.integers(0, len(pop), k)
    best_idx = idxs[np.argmax(fitness[idxs])]
    return pop[best_idx]

def crossover(p1, p2, rng):
    """One-point crossover."""
    point = rng.integers(1, len(p1))
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(child, rng):
    mask = rng.random(len(child)) < MUT_RATE
    child[mask] += rng.normal(0, MUT_STDEV, np.sum(mask))
    return child


def run_one_seed(seed):
    """
    Running GeneticAlgorithm once (with a fixed random seed) and return:
      - Best_fitness_per_generation: list of floats, length = GENERATIONS
      - Best_overall: one float
    """
    rng = np.random.default_rng(seed)
    pop = init_pop_vec(rng, POP_SIZE)
    fitness = np.array([rollout_fitness(ind) for ind in pop])
    best_per_gen = []

    for g in range(GENERATIONS):
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fitness, rng)
            p2 = tournament_selection(pop, fitness, rng)
            c1, c2 = crossover(p1, p2, rng)
            c1 = mutate(c1.copy(), rng)
            c2 = mutate(c2.copy(), rng)
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:POP_SIZE])
        fitness = np.array([rollout_fitness(ind) for ind in pop])
        gen_best = np.max(fitness)
        best_per_gen.append(gen_best)

        # Quick progress print
        print(f"[seed {seed}] gen {g+1:02d}/{GENERATIONS}  best={gen_best:.4f} m")

    best_overall = np.max(fitness)  # global best
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
    plt.title("Experiment 1 — Genetic Algorithm on Gecko (BoxyRugged)")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/exp1_ga_student.png", dpi=160)  # Save one clean figure
    plt.show()


if __name__ == "__main__":
    main()

