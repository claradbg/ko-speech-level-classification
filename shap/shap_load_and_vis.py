import pickle
import shap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


label2id = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}


def shap_library_graph(shap_values, model, level):
    class_idx = label2id[level]

    matplotlib.rcParams['font.family'] = 'Noto Sans KR'
    plt.title(f"SHAP Token Contributions for Speech Level {level}")

    shap.plots.bar(shap_values[:, :, class_idx].mean(0), max_display=15, show=False)
    plt.savefig(f"figures/mean_{model}_sp_{level}_lib.png")


def shap_graph_manual(shap_values, model, level):
    class_idx = label2id[level]

    # Collect token contributions
    token_contributions = {}
    for example in shap_values:
        tokens = example.data
        values = example.values[:, class_idx]
        for token, val in zip(tokens, values):
            token_contributions.setdefault(token, []).append(val)


    # Average SHAP values per token
    avg_contributions = {token: np.mean(vals) for token, vals in token_contributions.items()}
    sorted_tokens = sorted(avg_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    tokens, values = zip(*sorted_tokens)

    # Create colors based on value sign
    colors = ['red' if val < 0 else 'blue' for val in values]

    # Plot
    #matplotlib.rcParams['font.family'] = 'Noto Sans KR'
    matplotlib.rcParams['font.family'] = 'NanumGothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 4))
    plt.bar(tokens, values, color=colors)
    plt.title(f"SHAP Token Contributions for Speech Level {level}", fontsize=16)
    plt.xticks(rotation=45, fontsize=14)
    plt.xlabel("Token", fontsize=13)
    plt.ylabel("SHAP Value", fontsize=13)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"figures/mean_{model}_sp_{level}_manual.png")


def load_shap(model, level):
    with open(f"shap_outputs/{model}_shap_values_{level}.pkl", "rb") as f:
        loaded_shap_values = pickle.load(f)
        return loaded_shap_values

# Speech levels to analyze
speech_levels = [1, 2, 4, 5, 6]
# speech_levels = [5, 6]
models = ["multi", "kobert"]
#models = ["kobert"]

# SHAP plot for each speech level
for level in speech_levels:
    for model in models:
        # Get SHAP values
        shap_values = load_shap(model, level)

        shap_library_graph(shap_values, model, level)
        plt.clf()
        shap_graph_manual(shap_values, model, level)
        plt.clf()
