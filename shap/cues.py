import pickle
import numpy as np
import pandas as pd


label2id = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}

def load_shap(model, level):
    with open(f"shap_outputs/{model}_shap_values_{level}.pkl", "rb") as f:
        loaded_shap_values = pickle.load(f)
        return loaded_shap_values


def avg_shap(shap_values, level):
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
    sorted_tokens = sorted(avg_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:100]
    tokens, values = zip(*sorted_tokens)

    return tokens, values

def add_to_summary(tokens, values, level, model, df):
    total = len(tokens)
    for i in range(len(tokens)):
        token = tokens[i]
        value = values[i]

        df = pd.concat([pd.DataFrame([[level, model, token, value, ""]], columns=df.columns), df], ignore_index=True)

        print(f"{i/total}% done")

    return df


# Speech levels to analyze
speech_levels = [1, 2, 4, 5, 6]
# speech_levels = [5, 6]
models = ["multi", "kobert"]
# models = ["kobert"]

df = pd.DataFrame(columns=["Speech Level", "Model", "Token", "SHAP Value", "Cue Type"])

for level in speech_levels:
    for model in models:
        print()
        print(f"Adding SL {level} of model {model}")

        # Get SHAP values
        shap_values = load_shap(model, level)

        tokens, values = avg_shap(shap_values, level)
        df = add_to_summary(tokens, values, level, model, df)
        


df.to_csv("shap_cue_summary_top100.csv")