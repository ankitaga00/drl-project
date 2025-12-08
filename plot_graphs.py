import matplotlib.pyplot as plt


def read_metrics(filename):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        data = {}

        for line in lines:
            key, value = line.strip().split("=")
            data[key] = float(value)

        return data
    except:
        print(f"⚠ WARNING: Could not read {filename}")
        return None


# Load metric dictionaries
fixed = read_metrics("results_fixed.txt")
independent = read_metrics("results_independent.txt")
colight = read_metrics("results_colight.txt")

methods = ["Fixed-Time", "Independent RL", "CoLight"]

# Choose which metric to plot: reward / switches / queue
metric_key = "avg_reward"  # 👈 change to avg_queue or avg_switches

metric_values = [
    fixed[metric_key] if fixed else None,
    independent[metric_key] if independent else None,
    colight[metric_key] if colight else None,
]

# Plot
plt.figure(figsize=(8, 4))
bars = plt.bar(methods, metric_values)

for bar, val in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{val:.1f}",
        ha="center",
        va="bottom"
    )

plt.title(f"Comparison of {metric_key.replace('_', ' ').title()} Across Methods")
plt.ylabel(metric_key.replace("_", " ").title())
plt.grid(axis='y', alpha=0.3)
plt.savefig("comparison_plot.png", dpi=300)
plt.show()
