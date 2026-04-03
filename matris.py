# matris.py
# Multi-label (17 SDG) için görsel confusion matrix + metrik raporu üretir.
# Çıktılar:
# - confusion_png/SDGx_confusion.png (her sınıf için 2x2 heatmap)
# - confusion_all_labels_grid.png (tüm sınıflar tek sayfada)
# - classification_report.txt (sınıf bazlı precision/recall/f1)
#
# Not: Multi-label'da klasik 17x17 confusion matrix doğru değildir.
#      Bunun yerine her sınıf için 2x2 (TN,FP,FN,TP) confusion kullanılır.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================
# 1) CONFIG (YOLLARI KENDİNE GÖRE GÜNCELLE)
# =========================
class Config:
    MODEL_NAME = "allenai/scibert_scivocab_uncased"

    # Excel dosyanın yolu
    FILE_PATH = r"C:\Users\HP\OneDrive - erbakan.edu.tr\Masaüstü\suggester_guncel\TF_SDG.xlsx"

    # Kaydettiğin modelin yolu
    MODEL_PATH = r"C:\Users\HP\OneDrive - erbakan.edu.tr\Masaüstü\suggester_guncel\scibert_final_model (1).pt"

    # (Opsiyonel) threshold vektörü varsa aynı klasöre koy: scibert_thresholds.npy
    THR_PATH = r"C:\Users\HP\OneDrive - erbakan.edu.tr\Masaüstü\suggester_guncel\scibert_thresholds.npy"

    MAX_LEN = 512
    BATCH_SIZE = 8
    SEED = 42

    # Etiket kolon isimleri (Excel'de birebir böyle olmalı)
    SDG_LABELS = [f"SDG{i}" for i in range(1, 18)]
    NUM_LABELS = len(SDG_LABELS)

    # Excel'deki metin kolonları
    TITLE_COL = "title"
    ABSTRACT_COL = "abstract"

    # Split oranı (val set için)
    VAL_SIZE = 0.15

# =========================
# 2) DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# 3) DATASET
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
            "input_ids": enc["input_ids"].squeeze(0),          # (T,)
            "attention_mask": enc["attention_mask"].squeeze(0),# (T,)
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)  # (17,)
        }

def prepare_val_split():
    if not os.path.exists(Config.FILE_PATH):
        raise FileNotFoundError(f"Excel bulunamadı: {Config.FILE_PATH}")

    df = pd.read_excel(Config.FILE_PATH, engine="openpyxl")

    # Kolon kontrolü (hata mesajı anlaşılır olsun)
    need_cols = [Config.TITLE_COL, Config.ABSTRACT_COL] + Config.SDG_LABELS
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Excel'de eksik kolon(lar) var: {missing}\nMevcut kolonlar: {list(df.columns)}")

    texts = []
    for _, row in df.iterrows():
        t = str(row.get(Config.TITLE_COL, "")).strip()
        a = str(row.get(Config.ABSTRACT_COL, "")).strip()
        texts.append(f"{t} {a}".lower())

    labels = df[Config.SDG_LABELS].values.astype(float)

    # Sadece val kısmını kullanacağız (confusion için)
    _, X_val, _, y_val = train_test_split(
        texts, labels,
        test_size=Config.VAL_SIZE,
        random_state=Config.SEED
    )
    return X_val, y_val

# =========================
# 4) MODEL
# =========================
class SciBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # safetensors kullanmaya zorla (torch<2.6 kısıtını aşmak için genelde işe yarar)
        self.bert = AutoModel.from_pretrained(
            Config.MODEL_NAME,
            use_safetensors=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, Config.NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]               # (B,768)
        logits = self.classifier(self.dropout(cls))        # (B,17)
        return logits

# =========================
# 5) VISUALIZATION HELPERS
# =========================
def save_per_label_png(mcm, out_dir="confusion_png"):
    os.makedirs(out_dir, exist_ok=True)
    for i, lab in enumerate(Config.SDG_LABELS):
        cm = mcm[i]  # [[TN, FP],[FN, TP]]

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm)

        ax.set_title(lab)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{lab}_confusion.png"), dpi=200)
        plt.close(fig)

    print(f"PNG'ler kaydedildi: {out_dir}/")

def save_grid_png(mcm, out_path="confusion_all_labels_grid.png", cols=5):
    n = len(Config.SDG_LABELS)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for i, lab in enumerate(Config.SDG_LABELS):
        ax = axes[i]
        cm = mcm[i]
        im = ax.imshow(cm)

        ax.set_title(lab, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center", fontsize=9)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    print(f"Grid görsel kaydedildi: {out_path}")

# =========================
# 6) MAIN
# =========================
def main():
    set_seed(Config.SEED)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model bulunamadı: {Config.MODEL_PATH}")

    # Modeli yükle
    model = SciBERTClassifier().to(DEVICE)

    # FutureWarning'i azaltmak için weights_only=True kullanıyoruz (PyTorch destekliyorsa)
    try:
        state = torch.load(Config.MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        # Eski torch sürümlerinde weights_only parametresi olmayabilir
        state = torch.load(Config.MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state)
    model.eval()

    # Threshold vektörü (opsiyonel)
    thr_vec = None
    if os.path.exists(Config.THR_PATH):
        thr_vec = np.load(Config.THR_PATH)
        if thr_vec.shape[0] != Config.NUM_LABELS:
            print("Uyarı: Threshold vektör boyutu 17 değil, global 0.5 kullanılacak.")
            thr_vec = None
        else:
            print("Threshold vector loaded:", Config.THR_PATH)
    else:
        print("Threshold vector not found, using global 0.5")

    # Val set hazırlığı
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
    val_texts, val_labels = prepare_val_split()
    val_ds = SDGDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Tahminleri topla
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

    all_probs = np.vstack(all_probs)      # (N,17)
    all_targets = np.vstack(all_targets)  # (N,17)

    # 0/1 karar
    if thr_vec is not None:
        y_pred = (all_probs > thr_vec.reshape(1, -1)).astype(int)
    else:
        y_pred = (all_probs > 0.5).astype(int)

    # Confusion
    mcm = multilabel_confusion_matrix(all_targets, y_pred)

    # Konsola yazdır (isteğe bağlı)
    print("\n--- Per-label Confusion (TN FP FN TP) ---")
    for i, lab in enumerate(Config.SDG_LABELS):
        tn, fp, fn, tp = mcm[i].ravel()
        print(f"{lab}: TN={tn} FP={fp} FN={fn} TP={tp}")

    TN = mcm[:, 0, 0].sum()
    FP = mcm[:, 0, 1].sum()
    FN = mcm[:, 1, 0].sum()
    TP = mcm[:, 1, 1].sum()
    print("\n--- MICRO TOTAL ---")
    print(f"TN={TN} FP={FP} FN={FN} TP={TP}")

    # Görselleri kaydet
    save_per_label_png(mcm, out_dir="confusion_png")
    save_grid_png(mcm, out_path="confusion_all_labels_grid.png", cols=5)

    # Classification report dosyaya yaz
    report = classification_report(all_targets, y_pred, target_names=Config.SDG_LABELS, zero_division=0)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nclassification_report.txt kaydedildi.")
    print("Bitti.")

if __name__ == "__main__":
    main()
