## ANOVA STATISTICAL TEST ##
import scipy.stats as stats
import wandb
from pathlib import Path
import pandas as pd

# Same directory
from params import ENTITY, PROJECT, ALGOS, SEEDS


def main():
    # Start an analysis run (optional job_type)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="Analysis",
        job_type="analysis",
        reinit=True,
        config={"Experiment": "Analysis"},
    )

    algo_best_per_run = {algo: [] for algo in ALGOS}  # collect 1 value per run

    for algo in ALGOS:
        for seed in SEEDS:
            # Use the artifact (downloads locally)
            artifact_name = f"{algo}-seed{seed}-raw-data:latest"
            artifact = run.use_artifact(f"{ENTITY}/{PROJECT}/{artifact_name}")

            # Read the artifact's parquet file
            artifact_dir = artifact.download()  # returns local path
            expected_file = Path(artifact_dir) / f"{algo}-seed{seed}_raw.parquet"
            df = pd.read_parquet(expected_file)

            all_f = []

            for _, generation in df.iterrows():  # collect best fitness
                f_vals = generation["fitness"]
                all_f.extend(f_vals)
                best_overall = max(all_f)
                algo_best_per_run[algo].append(best_overall)
    
    # Final fitness per seed
    baseline_runs = algo_best_per_run["Baseline"]
    ga_runs       = algo_best_per_run["GA"]
    cmaes_runs    = algo_best_per_run["CMA-ES"]

    # Run One-Way ANOVA
    f_statistic, p_value = stats.f_oneway(baseline_runs, ga_runs, cmaes_runs)

    # Log results to Weights & Biases
    run.log({
        "F-statistic": f_statistic,
        "P-value": p_value,
    })

    # Output results
    print("One-Way ANOVA on Final Fitness Scores")
    print("-------------------------------------")
    print(f"F-statistic: {f_statistic:.5f}")
    print(f"P-value:     {p_value:.5f}")

    # Interpretation
    if p_value < 0.05:
        print("Significant difference found between at least two groups.")
    else:
        print("No significant difference found.")
    
    # Close analysis run
    run.finish()


if __name__ == "__main__":
    main()
