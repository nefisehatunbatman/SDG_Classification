# treshold.py  (esnek threshold vektörü üretir ve kaydeder)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# =========================
# CONFIG (YOLLARI KENDİNE GÖRE KONTROL ET)
# =========================
class Config:
    MODEL_NAME = "allenai/scibert_scivocab_uncased"
    FILE_PATH  = r"C:\Users\HP\OneDrive - erbakan.edu.tr\Masaüstü\suggester_guncel\TF_SDG.xlsx"
    MODEL_PATH = r"C:\Users\HP\OneDrive - erbakan.edu.tr\Masaüstü\suggester_guncel\scibert_final_model (1).pt"

    MAX_LEN = 512
    BATCH_SIZE = 8
    SEED = 42
    VAL_SIZE = 0.15

    SDG_LABELS = [f"SDG{i}" for i in range(1, 18)]
    NUM_LABELS = len(SDG_LABELS)

    TITLE_COL = "title"
    ABSTRACT_COL = "abstract"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# =========================
# DATASET
# =========================
class SDGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=Config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

def prepare_val_split():
    df = pd.read_excel(Config.FILE_PATH, engine="openpyxl")

    need_cols = [Config.TITLE_COL, Config.ABSTRACT_COL] + Config.SDG_LABELS
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Excel'de eksik kolon(lar): {missing}\nMevcut kolonlar: {list(df.columns)}")

    texts = []
    for _, row in df.iterrows():
        t = str(row.get(Config.TITLE_COL, "")).strip()
        a = str(row.get(Config.ABSTRACT_COL, "")).strip()
        texts.append(f"{t} {a}".lower())

    labels = df[Config.SDG_LABELS].values.astype(float)

    _, X_val, _, y_val = train_test_split(
        texts, labels, test_size=Config.VAL_SIZE, random_state=Config.SEED
    )
    return X_val, y_val

# =========================
# MODEL
# =========================
class SciBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME, use_safetensors=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, Config.NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

# =========================
# THRESHOLD OPT
# =========================
def main():
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model yok: {Config.MODEL_PATH}")

    # model yükle
    model = SciBERTClassifier().to(DEVICE)
    try:
        state = torch.load(Config.MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(Config.MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # data
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
    val_texts, val_labels = prepare_val_split()
    val_loader = DataLoader(SDGDataset(val_texts, val_labels, tokenizer),
                            batch_size=Config.BATCH_SIZE, shuffle=False)

    # probs + targets
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].cpu().numpy()

            logits = model(ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(y)

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    # threshold bul
    best_thresholds = []
    for i in range(Config.NUM_LABELS):
        y_true = all_targets[:, i].astype(int)
        p = all_probs[:, i]

        best_t = 0.5
        best_f1 = -1.0

        for t in np.arange(0.05, 0.95, 0.05):
            y_pred = (p > t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        best_thresholds.append(best_t)

    best_thresholds = np.array(best_thresholds, dtype=np.float32)
    np.save("scibert_thresholds.npy", best_thresholds)

    print("\nKaydedildi: scibert_thresholds.npy")
    print("İlk 5 threshold:", best_thresholds[:5])
    print("Tüm thresholdlar:", best_thresholds)

if __name__ == "__main__":
    main()
