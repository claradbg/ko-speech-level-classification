from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import Dataset
import pandas as pd
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

with open("label_mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)
    label2id = {int(float(k)): int(float(v))
                for k, v in mappings["label2id"].items()}
    id2label = {int(float(k)): int(float(v))
                for k, v in mappings["id2label"].items()}

# Load data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()
val_texts = val_df["text"].tolist()
val_labels = val_df["label"].tolist()
test_texts = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

# Load tokenizer
model_checkpoint = "Genius1237/xlm-roberta-large-tydip"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["Subtitle"], truncation=True, padding="max_length", max_length=128)


# Prepare dataset
train_data = {"Subtitle": train_texts, "label": train_labels}
val_data = {"Subtitle": val_texts, "label": val_labels}
test_data = {"Subtitle": test_texts, "label": test_labels}

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


num_labels = 5  # how many labels the model should expect
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=35,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

print("Evaluation:")
test_metrics = trainer.evaluate(
    eval_dataset=test_dataset, metric_key_prefix="test")
print(test_metrics)


# save and synch model and tokenizer to avoid mismatches in input ID
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

# evaluate on test dataset
test_results = trainer.evaluate()

# save results to JSON
with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(test_results, f, indent=4, ensure_ascii=False)
