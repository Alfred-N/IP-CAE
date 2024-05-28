import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Load the CSV file
csv_file_path = "logs/log_17_37_25_885853__28_05_24/train_log.csv"  # Update this path
df = pd.read_csv(csv_file_path)

# Filter the DataFrame to get the relevant metric
metric = "grad_norm_logits"
metric_data = df[["epoch", "step", metric]].dropna()

# Plot the metric
fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

sns.lineplot(data=metric_data, x="step", y=metric, ax=ax, label=metric)

ax.set_xlabel("Step")
ax.set_ylabel(
    r"$\|  (\log \boldsymbol{\alpha}_{t+1} - \log \boldsymbol{\alpha}_{t}) \cdot 1/\eta \|_2$"
)
ax.set_title(f"{metric} over Steps")
sns.move_legend(ax, "lower right")

# Save the plot
output_dir = "path_to_output_directory"  # Update this path
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, f"{metric}.svg"))
plt.show()
