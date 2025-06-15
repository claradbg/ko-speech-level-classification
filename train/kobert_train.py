import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from kobert_tokenizer import KoBERTTokenizer
from datasets import Dataset
#from konlpy.tag import Okt
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load label mappings
with open("label_mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)
    label2id = {int(float(k)): int(float(v))
                for k, v in mappings["label2id"].items()}
    id2label = {int(float(k)): int(float(v))
                for k, v in mappings["id2label"].items()}

# Load your dataset
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()
val_texts = val_df["text"].tolist()
val_labels = val_df["label"].tolist()
test_texts = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

# Initialize KoNLPy tokenizer (Okt)
#okt = Okt()

# Morphological preprocessing function
# def preprocess_morphs(text):
#     morphs = okt.morphs(str(text))
#     return " ".join(morphs)


# Apply morphological analysis to all datasets
# train_df["text_morphed"] = train_df["text"].apply(preprocess_morphs)
# val_df["text_morphed"] = val_df["text"].apply(preprocess_morphs)
# test_df["text_morphed"] = test_df["text"].apply(preprocess_morphs)



# Load KoBERT tokenizer and model
kobert_model = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(kobert_model, use_fast=False)
model = BertForSequenceClassification.from_pretrained(
    kobert_model,
    num_labels=5,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Tokenization function
def tokenize_function(example):
    return tokenizer(
        example["Subtitle"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# # Prepare Hugging Face datasets
# train_dataset = Dataset.from_pandas(train_df[["Subtitle", "label"]])
# val_dataset = Dataset.from_pandas(val_df[["Subtitle", "label"]])
# test_dataset = Dataset.from_pandas(test_df[["Subtitle", "label"]])

# Prepare dataset
train_data = {"Subtitle": train_texts, "label": train_labels}
val_data = {"Subtitle": val_texts, "label": val_labels}
test_data = {"Subtitle": test_texts, "label": test_labels}


# Tokenize
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# val_dataset = val_dataset.map(tokenize_function, batched=True)
# test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = Dataset.from_dict(train_data).map(
    tokenize_function, batched=True)
val_dataset = Dataset.from_dict(val_data).map(tokenize_function, batched=True)
test_dataset = Dataset.from_dict(test_data).map(
    tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=[
                         "input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=[
                       "input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=[
                        "input_ids", "attention_mask", "label"])

# Define metrics


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# Training arguments
training_args = TrainingArguments(
    output_dir="./kobert_results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=35,
    weight_decay=0.01,
    logging_dir="./kobert_logs",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate
print("Evaluation on test set:")
test_metrics = trainer.evaluate(
    eval_dataset=test_dataset, metric_key_prefix="test")
print(test_metrics)

# Save model and tokenizer
model.save_pretrained("kobert_saved_model")
tokenizer.save_pretrained("kobert_saved_model")

# Save results
with open("kobert_test_results.json", "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=4, ensure_ascii=False)
