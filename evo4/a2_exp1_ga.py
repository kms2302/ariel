# =========================
# EXPERIMENT 1 (GeneticAlgorithm)
# Task: Making the gecko walk as far as possible in the environment BoxyRugged.
# Fitness = XY displacement of the "core" body after a fixed number of steps.
# =========================

import math
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import wandb
from pathlib import Path
import pandas as pd

# Same directory
from utils import (
    NUM_JOINTS, 
    rollout_fitness, 
    moving_average, 
    init_param_vec,
    init_pop_vec,
)
from a2_params import (
    GENERATIONS,
    POP_SIZE,
    SEEDS,
    MUT_RATE,
    MUT_STDEV,
    ENTITY,
    PROJECT,
    CONFIG,
)


def tournament_selection(pop, fitness, rng, k=3):
    """Pick one parent using k-way tournament."""
    idxs = rng.integers(0, len(pop), k)             # Choosing k random indices
    best_idx = idxs[np.argmax(fitness[idxs])]       # Picking the best candidate among them
    return pop[best_idx]

def crossover(p1, p2, rng):
    """One-point crossover."""
    point = rng.integers(1, len(p1))                # picking a random point != 0
    c1 = np.concatenate([p1[:point], p2[point:]])   # child 1
    c2 = np.concatenate([p2[:point], p1[point:]])   # child 2
    return c1, c2

def mutate(child, rng):
    mask = rng.random(len(child)) < MUT_RATE                # Mask marking which genes will be mutated
    child[mask] += rng.normal(0, MUT_STDEV, np.sum(mask))   # Applying noise where mask=True
    return child


def main():
    ALGO = "GA"

    for seed in SEEDS:
        run_name = f"{ALGO}-seed{seed}"
        
        # Start a new wandb run to track this script.
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=run_name,
            config=CONFIG,
        )
        wandb.config.update({
            "Experiment": ALGO,
            "Seed": seed,
            "Mutation Rate": MUT_RATE,
            "Mutation SD": MUT_STDEV,
        })

        rng = np.random.default_rng(seed)
        pop = init_pop_vec(rng, POP_SIZE)
        gen_fitness = np.array([rollout_fitness(ind) for ind in pop])
        best_per_gen = []
        best_overall = -np.inf
        generations = []

        for g in range(GENERATIONS):
            new_pop = []

            while len(new_pop) < POP_SIZE:
                # Selecting 2 parents
                p1 = tournament_selection(pop, gen_fitness, rng)
                p2 = tournament_selection(pop, gen_fitness, rng)

                # Performing crossover
                c1, c2 = crossover(p1, p2, rng)

                # Applying mutation to children
                c1 = mutate(c1.copy(), rng)
                c2 = mutate(c2.copy(), rng)
                new_pop.extend([c1, c2])

            # Replacing old population
            pop = np.array(new_pop[:POP_SIZE])
            gen_fitness = np.array([rollout_fitness(ind) for ind in pop])
            
            # For plotting: keeping the best (i.e., max displacement) this generation
            best_in_gen = np.max(gen_fitness)
            best_per_gen.append(best_in_gen)

            # Update overall best across generations
            best_overall = max(best_overall, best_in_gen)

            # Log this gen (i.e., step) to Weights & Biases
            run.log({
                "gen": g,
                "Best fitness in generation (BoxyRugged gecko)": best_in_gen, 
                "Best fitness across generations (BoxyRugged gecko)": best_overall,
            }, step=g)

            # Append raw rows for this generation
            generations.append({
                "gen": g,
                "fitness": gen_fitness,
            })

        # End of run: create a DataFrame and write to Parquet (or CSV)
        out_dir = Path("wandb_artifacts")
        out_dir.mkdir(exist_ok=True)
        file_path = out_dir / f"{run_name}_raw.parquet"
        df = pd.DataFrame(generations)
        df.to_parquet(file_path, index=False)

        # Create an artifact, add the file, and log it
        artifact = wandb.Artifact(
            name=f"{run_name}-raw-data",
            type="raw_data",
            metadata={"generations": 30, "num_rows": len(df)}
        )
        artifact.add_file(str(file_path))
        run.log_artifact(artifact)
        
        # Finish the run and upload any remaining data.
        run.finish()


if __name__ == "__main__":
    main()
