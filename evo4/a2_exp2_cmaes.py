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
import wandb
from pathlib import Path
import pandas as pd

# Same directory
from utils import NUM_JOINTS, rollout_fitness, moving_average, init_param_vec
from a2_params import (
    GENERATIONS,
    POP_SIZE,
    SEEDS,
    SIGMA_INIT,
    ENTITY,
    PROJECT,
    CONFIG,
)


def main():
    ALGO = "CMA-ES"  # Weights & Biases configuration

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
            "Initial Sigma": SIGMA_INIT,
        })

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
        best_overall = -np.inf
        generations = []

        for g in range(GENERATIONS):
            # ask(): sampling a whole population of candidate solutions
            population = es.ask()

            # Evaluating everyone; we negate because CMA-ES expects a "loss"
            losses = []
            gen_fitness = []
            for x in population:
                f = rollout_fitness(np.asarray(x, dtype=np.float64))  # Displacement (bigger is better)
                losses.append(-f)  # Negate. Lower is better for CMA
                gen_fitness.append(f)

            # tell(): giving CMA-ES the evaluated losses so it can update its search distribution
            es.tell(population, losses)

            # For plotting: keeping the best (i.e., max displacement) this generation
            best_in_gen = max(gen_fitness)
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
