import matplotlib.pyplot as plt

def read_value(filename):
    try:
        with open(filename, "r") as f:
            val = float(f.read().strip())
        return val
    except:
        print(f"⚠ WARNING: File {filename} not found or unreadable.")
        return None

# Read logged averages
fixed = read_value("results_fixed.txt")
independent = read_value("results_independent.txt")
colight = read_value("results_colight.txt")

labels = []
values = []

if fixed is not None:
    labels.append("Fixed-time")
    values.append(fixed)

if independent is not None:
    labels.append("Independent RL")
    values.append(independent)

if colight is not None:
    labels.append("CoLight")
    values.append(colight)

# Plot
plt.figure(figsize=(8,4))
bars = plt.bar(labels, values)

# Annotate bars
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{val:.1f}", ha="center", va="bottom")

plt.ylabel("Average Total Reward (lower is better)")
plt.title("Traffic Signal Control Comparison")
plt.grid(axis='y', alpha=0.3)

plt.savefig("comparison_plot.png", dpi=300)
plt.show()
