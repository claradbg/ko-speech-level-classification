import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import classification_report
import pandas as pd
from datasets import Dataset

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("saved_model")
tokenizer = AutoTokenizer.from_pretrained("saved_model")

# Load test set
test_df = pd.read_csv("test.csv")
test_texts = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

# Prepare dataset
test_data = {"Subtitle": test_texts, "label": test_labels}

# Tokenize
def tokenize_function(example):
    return tokenizer(
        example["Subtitle"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_dataset = Dataset.from_dict(test_data).map(tokenize_function, batched=True)

# Set format
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Predict
training_args = TrainingArguments(output_dir="unused_dir")  # minimal config
trainer = Trainer(model=model, args=training_args)
predictions = trainer.predict(test_dataset)

logits, labels = predictions.predictions, predictions.label_ids
preds = logits.argmax(-1)

# Load label map
import json
with open("label_mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)
    id2label = {int(float(k)): str(v) for k, v in mappings["id2label"].items()}

# Print classification report
from sklearn.metrics import classification_report
label_ids = sorted(id2label.keys())
print(classification_report(
    labels, preds,
    labels=label_ids,
    target_names=[id2label[i] for i in label_ids],
    zero_division=1
))


from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

# Generate multilabel confusion matrix
conf_matrices = multilabel_confusion_matrix(labels, preds, labels=sorted(id2label.keys()))

# Save confusion matrices per class
with open("multi_confusion_matrices.txt", "w", encoding="utf-8") as f:
    for i, cm in enumerate(conf_matrices):
        label_name = id2label[i]
        f.write(f"Class: {label_name} (label {i})\n")
        f.write(np.array2string(cm))
        f.write("\n\n")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(labels, preds, labels=sorted(id2label.keys()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[i] for i in sorted(id2label.keys())])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.tight_layout()
plt.savefig("multi_confusion_matrix.png", dpi=300)