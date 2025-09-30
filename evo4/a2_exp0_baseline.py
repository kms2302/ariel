# =========================
## EXPERIMENT 0 (BASELINE)
# Fitness = XY displacement of the "core" body after a fixed number of steps.
# =========================

import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
import pandas as pd

# Same directory
from utils import rollout_fitness, moving_average, NUM_JOINTS
from params import (
    GENERATIONS,
    SEEDS,
    POP_SIZE,
    ENTITY,
    PROJECT,
    CONFIG,
)


def main():
    """
    Running the baseline (random search).
    In each generation, POP_SIZE random solutions are generated,
    and the best one is kept.
    """
    ALGO = "Baseline"

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
        })

        rng = np.random.default_rng(seed)
        best_per_gen = []
        best_overall = -np.inf
        generations = []

        for g in range(GENERATIONS):
            # Generate POP_SIZE random solutions per generation
            gen_fitness = []
            for _ in range(POP_SIZE):
                x = rng.normal(size=NUM_JOINTS * 2 + 1)
                f = rollout_fitness(x)
                gen_fitness.append(f)

            # Best fitness found in this generation
            best_in_gen = max(gen_fitness)

            # Update overall best across generations
            best_overall = max(best_overall, best_in_gen)
            best_per_gen.append(best_in_gen)

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
