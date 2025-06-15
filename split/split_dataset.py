# loading dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import json

# load csv
df = pd.read_csv('thesis_data.csv')
df = df.drop(columns=["Time", "Translation", "Transliteration", "K-Drama Title", "Year", "Episode", "Cue", "Cue Type"])

df = df.dropna() # remove NA values
#print(df)

print(f"Total rows: {len(df)}")

# print(df.head())

# these are the 5 levels we care about, respectively: 합쇼체, 해요체, 하게체, 반말체, 해라체
df = df[df["Speech Level"].isin([1, 2, 4, 5, 6])]
df = df.rename(columns={"Speech Level": "label"})

# create mappings between original speech levels and internal label IDs
unique_levels = sorted(df['label'].unique())  # [1, 2, 4, 5, 6]
label2id = {level: idx for idx, level in enumerate(unique_levels)} # e.g., {1:0, 2:1, 4:2, 5:3, 6:4}. normally counting starts at 0
id2label = {idx: level for level, idx in label2id.items()} # inverse mapping

# create a new column 'label_id' to be used for training
df["label_id"] = df["label"].map(label2id)

# train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Subtitle"].tolist(), 
    df["label_id"].tolist(), 
    test_size=0.2, 
    random_state=42,
    stratify=df["label_id"].tolist()
)

# test-validation split
test_texts, val_texts, test_labels, val_labels = train_test_split(
    val_texts, 
    val_labels, 
    test_size=0.5, 
    random_state=42,
    stratify=val_labels
)

# Save to files
pd.DataFrame({"text": train_texts, "label": train_labels}).to_csv("train.csv", index=False)
pd.DataFrame({"text": val_texts, "label": val_labels}).to_csv("val.csv", index=False)
pd.DataFrame({"text": test_texts, "label": test_labels}).to_csv("test.csv", index=False)

# verify output
print("Label2ID mapping:", label2id)
print("ID2Label mapping:", id2label)
print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}, Test size: {len(test_texts)}")
print(f"Samples in train set per class     : {Counter(train_labels).items()}")
print(f"Samples in test set per class      : {Counter(test_labels).items()}")
print(f"Samples in validation set per class: {Counter(val_labels).items()}")

with open("label_mappings.json", "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=4)
