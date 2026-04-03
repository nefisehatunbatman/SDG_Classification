import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm.notebook import tqdm
import re
import os

# --- AYARLAR ---
MODEL_PATH = "/kaggle/working/best_model_ingilizce.pt"
PREDICT_FILE = "/kaggle/input/sdg-dataset/TF_SDG.xlsx" 
OUTPUT_FILE = "FINAL_OPTIMIZE_SONUC.xlsx"
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🔥 EN İYİ EŞİK LİSTESİ (Senin Bulduğun)
OPTIMIZED_THRESHOLDS = [
    0.50, 0.20, 0.60, 0.40, 0.25, 0.25, 0.60, 0.40, 0.30, 
    0.25, 0.20, 0.15, 0.50, 0.45, 0.35, 0.50, 0.35
]

# --- MODEL (AYNI) ---
class XLMRobertaMultiLabel(nn.Module):
    def __init__(self, model_name, n_labels):
        super().__init__()
        self.xlm_roberta = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 384), nn.ReLU(), nn.BatchNorm1d(384), nn.Dropout(0.2), nn.Linear(384, n_labels)
        )
    def forward(self, ids, mask):
        o = self.xlm_roberta(ids, attention_mask=mask)
        return self.classifier(self.drop(o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:, 0]))

# --- YARDIMCILAR ---
FULL_SDG_LABELS = [f"SDG {i}" for i in range(1, 18)]
def smart_combine_text(row, columns):
    title_col = next((c for c in columns if str(c).lower() == 'title'), None)
    abstract_col = next((c for c in columns if str(c).lower() in ['abstract', 'abstract_en', 'description']), None)
    title = str(row[title_col]).strip() if title_col else ""
    abstract = str(row[abstract_col]).strip() if abstract_col else ""
    text = f"{title} {abstract}".strip()
    return re.sub(r"\s+", " ", text) if len(text) > 5 else "[empty_text]"

# --- FİNAL TAHMİN ---
def predict_final():
    print(f"📂 Model Yükleniyor...")
    model = XLMRobertaMultiLabel(MODEL_NAME, 17)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    print(f"📊 Dosya Okunuyor: {PREDICT_FILE}")
    if PREDICT_FILE.endswith('.xlsx'): df = pd.read_excel(PREDICT_FILE, engine="openpyxl")
    else: df = pd.read_csv(PREDICT_FILE)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cols = df.columns.tolist()
    texts = [smart_combine_text(row, cols) for _, row in df.iterrows()]
    
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    ds = torch.utils.data.TensorDataset(enc['input_ids'], enc['attention_mask'])
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    
    probs_list = []
    with torch.no_grad():
        for batch in tqdm(dl):
            ids, mask = [b.to(DEVICE) for b in batch]
            logits = model(ids, mask)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
            
    all_probs = np.vstack(probs_list)
    
    results = []
    for i in range(len(df)):
        probs = all_probs[i]
        active = []
        for j in range(17):
            if probs[j] > OPTIMIZED_THRESHOLDS[j]:
                active.append(FULL_SDG_LABELS[j])
        
        res = {"Smart Predictions": ", ".join(active) if active else "None"}
        top_idx = np.argmax(probs)
        res["Top1 SDG"] = FULL_SDG_LABELS[top_idx]
        res["Top1 Score"] = f"%{probs[top_idx]*100:.1f}"
        results.append(res)
        
    final_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    final_df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ FİNAL DOSYA HAZIR: {OUTPUT_FILE}")

if __name__ == "__main__":
    predict_final()