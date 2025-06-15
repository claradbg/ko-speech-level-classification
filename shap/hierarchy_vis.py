import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("shap_cue_summary_top100.csv")

# Group by Model and Cue Type: mean SHAP + count occurrences
cue_sums = df.groupby(["Model", "Cue Type"]).agg(
    total_shap=("SHAP Value", "mean"),
    count=("SHAP Value", "count")
).reset_index()

# Plot one graph per model
models = cue_sums["Model"].unique()

for model in models:
    model_data = cue_sums[cue_sums["Model"] == model].sort_values(by="total_shap", ascending=False)
    
    # Create custom x-axis labels with counts
    x_labels = [f"{cue}\n({cnt})" for cue, cnt in zip(model_data["Cue Type"], model_data["count"])]

    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, model_data["total_shap"], color="cornflowerblue", edgecolor="black")
    plt.title(f"Cue Type Hierarchy by Total SHAP Value — {model}", fontsize=13)
    plt.ylabel("Total SHAP Value")
    plt.xlabel("Cue Type (count)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()
