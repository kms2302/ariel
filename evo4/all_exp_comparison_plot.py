## COMPARISON OF BASELINE, GA, AND CMA-ES TOGETHER ##

import matplotlib.pyplot as plt
import numpy as np
import os
from utils import NUM_JOINTS, rollout_fitness, moving_average, init_param_vec

# Load the results from .npy files.
# The aformentioned results are the average fitness values across runs for each experiment.

results_dir = "__results__"
os.makedirs(results_dir, exist_ok=True)

baseline_mean = np.load(f"{results_dir}/exp0_baseline_mean.npy")
baseline_std = np.load(f"{results_dir}/exp0_baseline_std.npy")

ga_mean = np.load(f"{results_dir}/exp1_ga_mean.npy")
ga_std = np.load(f"{results_dir}/exp1_ga_std.npy")

cmaes_mean = np.load(f"{results_dir}/exp2_cmaes_mean.npy")
cmaes_std = np.load(f"{results_dir}/exp2_cmaes_std.npy")

# Create generations for raw baseline
generations = np.arange(1, len(baseline_mean) + 1)

# Plot raw baseline
plt.plot(generations, baseline_mean, label="Baseline", color="black")
plt.fill_between(generations, baseline_mean - baseline_std, baseline_mean + baseline_std,
                 color="black", alpha=0.2)

# Plot smoothed baseline with its own generations
smoothed = moving_average(baseline_mean, w=5)
smoothed_gens = np.arange(1, len(smoothed) + 1)

plt.plot(smoothed_gens, smoothed, label="Baseline (mean, moving avg w=5)", color="green")

# Plot GA
plt.plot(generations, ga_mean, label="GA (mean, moving avg w=5)", color="blue")
plt.fill_between(generations, ga_mean - ga_std, ga_mean + ga_std,
                 color="blue", alpha=0.2)

# Plot CMA-ES
plt.plot(generations, cmaes_mean, label="CMA-ES (mean, moving avg w=5)", color="orange")
plt.fill_between(generations, cmaes_mean - cmaes_std, cmaes_mean + cmaes_std,
                 color="orange", alpha=0.2)

# Labels and formatting
plt.title("Comparison of Baseline, GA, and CMA-ES on Gecko (BoxyRugged)")
plt.xlabel("Generation")
plt.ylabel("Fitness = XY displacement (m)")
plt.legend(loc="upper center",          # anchor the top center of the legend
    bbox_to_anchor=(0.5, -0.1),  # relative to the axes: center (0.5) and below (-0.1)
    ncol=2)
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(f"{results_dir}/all_exp_comparison.png", dpi=160)
plt.tight_layout()
plt.show()
