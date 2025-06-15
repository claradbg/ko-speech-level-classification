import pickle
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import shap
import pandas as pd
import torch.nn.functional as F

# Load model and tokenizer
model_path = "C:\\Users\\clari\\Code\\Thesis\\trained\\kobert_saved_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# SHAP-compatible prediction function
def predict_probabilities(texts):
    texts = list(map(str, texts))
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=150)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = F.softmax(outputs.logits, dim=1)
        #print(f"[DEBUG] Mean confidence: {probs.max(dim=1)[0].mean().item()}")

    return probs.numpy()

# Load data
df = pd.read_csv("C:\\Users\\clari\\Code\\Thesis\\trained\\thesis_data.csv")[["Subtitle", "Speech Level"]]


# Speech levels to analyze
speech_levels = [1, 2, 4, 5, 6]
#speech_levels = [4]
label2id = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}

# Initialize SHAP explainer once
explainer = shap.Explainer(predict_probabilities, masker=shap.maskers.Text(tokenizer))


def save_shap(shap_values, level):
    # Save SHAP values as pickle
    with open(f"shap_outputs/ko_shap_values_{level}.pkl", "wb") as f:
        pickle.dump(shap_values, f)


# SHAP plot for each speech level
for level in speech_levels:
    samples = df.loc[df["Speech Level"] == level, "Subtitle"].tolist()

    if len(samples) == 0:
        print(f"[WARNING] No samples for Speech Level {level}. Skipping.")
        continue

    # Get SHAP values (list-like Explanation)
    shap_values = explainer(samples)

    # Save the unified Explanation
    save_shap(shap_values, level)
    print(f"[INFO] SHAP values saved for level {level}")
