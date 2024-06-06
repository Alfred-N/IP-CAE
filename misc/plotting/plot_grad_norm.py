import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# Function to load and prepare data
def load_and_prepare_data(csv_file_path, metric, model_name):
    df = pd.read_csv(csv_file_path)
    metric_data = df[["epoch", "step", metric]].dropna()
    metric_data["model"] = model_name
    return metric_data


# Paths to CSV files for both models
csv_file_path_ipcae = (
    # "logs/log_ipcae_14_18_13_123784__29_05_24/train_log.csv"
    "logs/log_ipcae_10_56_15_302011__02_06_24/train_log.csv"
)
# csv_file_path_cae = "logs/log_cae_10_49_31_571900__02_06_24/train_log.csv"  # Set to None if no second model
csv_file_path_cae = None

# Metric to plot
# metric = "grad_norm_logits"
# metric = "avg_psi_t_dot_psi_t+1"
# metric = "avg_norm_psi_t_dot_psi_t+1"
metric = "W_prod_norm"

clean_metric_names = {
    "grad_norm_logits": r"$\|  (\log \boldsymbol{\alpha}_{t+1} - \log \boldsymbol{\alpha}_{t}) \cdot 1/\eta \|_2$",
    # "avg_psi_t_dot_psi_t+1": r"Avg. $\psi_i^t \cdot \psi_i^{t+1}$, $i=1,...,k$",
    "avg_norm_psi_t_dot_psi_t+1": r"Avg. $\| (\psi_i^t \cdot \psi_i^{t+1}) I \|_2$, $i=1,...,k$",
    "W_prod_norm": r"$\|WW^T\|_2$",
}

# clean_metric_names = {
#     "grad_norm_logits": r"Finite difference of $\log \boldsymbol{\alpha}$",
#     # "avg_psi_t_dot_psi_t+1": r"Avg. $\psi_i^t \cdot \psi_i^{t+1}$, $i=1,...,k$",
#     "avg_norm_psi_t_dot_psi_t+1": r"Avg. $\| (\psi_i^t \cdot \psi_i^{t+1}) \mathbf{I} \|_2$, $i=1,...,k$",
#     "W_prod_norm": r"$\|WW^T\|_2$",
# }

# Load and prepare data for the first model
metric_data_ipcae = load_and_prepare_data(csv_file_path_ipcae, metric, "IP-CAE")

if csv_file_path_cae:
    # Load and prepare data for the second model if provided
    metric_data_cae = load_and_prepare_data(csv_file_path_cae, metric, "CAE")
    # Combine the data from both models
    combined_metric_data = pd.concat([metric_data_ipcae, metric_data_cae])
else:
    combined_metric_data = metric_data_ipcae

# Plot the data
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)  # Adjusted figure size

sns.lineplot(data=combined_metric_data, x="step", y=metric, hue="model", ax=ax)

ax.set_xlabel("Step")
ax.set_ylabel(clean_metric_names[metric])
# title = f"{clean_metric_names[metric]} over Steps"
# if csv_file_path_cae:
#     title += " for IP-CAE and CAE"
# ax.set_title(title)
sns.move_legend(ax, "upper right")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

# Save the plot
output_dir = "misc/plotting/output"  # Update this path
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, f"{metric}.pdf"))
plt.show(block=True)
