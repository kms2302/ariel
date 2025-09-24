## ANOVA STATISTICAL TEST ##

import scipy.stats as stats

# Final fitness per seed (wrt to the results in the Overleaf pdf document)
baseline = [0.2318, 0.2123, 0.2256]
ga       = [0.1657, 0.2444, 0.1874]
cmaes    = [0.3388, 0.3571, 0.3194]

# Run One-Way ANOVA
f_statistic, p_value = stats.f_oneway(baseline, ga, cmaes)

# Output results
print("One-Way ANOVA on Final Fitness Scores")
print("-------------------------------------")
print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value:     {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("Significant difference found between at least two groups.")
else:
    print("No significant difference found.")

