import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm
import re
import os

# ====================================================================
# ⚙️ AYARLAR
# ====================================================================
MODEL_PATH = "/kaggle/working/best_model_ingilizce.pt"
VAL_FILE = "/kaggle/input/sdg-dataset/3dogrulama_orneklem.xlsx"
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FINAL_THRESHOLD = 0.50 

# ====================================================================
# 🏗️ MODEL SINIFI (Eğitim Koduyla BİREBİR AYNI)
# ====================================================================
class XLMRobertaMultiLabel(nn.Module):
    def __init__(self, model_name, n_labels):
        super().__init__()
        # 🔥 DÜZELTME 1: Burada 'xlm_roberta' ismini kullanmalıyız!
        self.xlm_roberta = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.3)
        
        # 🔥 DÜZELTME 2: Dosyada 'classifier' var, o yüzden bu isim kalmalı.
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.2),
            nn.Linear(384, n_labels)
        )

    def forward(self, ids, mask):
        # 🔥 DÜZELTME 3: Çağırırken de 'xlm_roberta' kullanıyoruz
        o = self.xlm_roberta(ids, attention_mask=mask)
        pooled_output = o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:, 0]
        return self.classifier(self.drop(pooled_output))

# ====================================================================
# 📥 YARDIMCI FONKSİYONLAR
# ====================================================================
FULL_SDG_LABELS = [
    "SDG 1: No Poverty", "SDG 2: Zero Hunger", "SDG 3: Good Health and Well-being",
    "SDG 4: Quality Education", "SDG 5: Gender Equality", "SDG 6: Clean Water and Sanitation",
    "SDG 7: Affordable and Clean Energy", "SDG 8: Decent Work and Economic Growth",
    "SDG 9: Industry, Innovation and Infrastructure", "SDG 10: Reduced Inequalities",
    "SDG 11: Sustainable Cities and Communities", "SDG 12: Responsible Consumption and Production",
    "SDG 13: Climate Action", "SDG 14: Life Below Water", "SDG 15: Life on Land",
    "SDG 16: Peace, Justice and Strong Institutions", "SDG 17: Partnerships for the Goals",
]
SDG_LABELS = [f"SDG{i}" for i in range(1, 18)]
NUM_LABELS = len(SDG_LABELS)

def read_table(path):
    if path.endswith('.xlsx'): return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)

# Akıllı Metin Birleştirici
def smart_combine_text(row, columns):
    title_col = next((c for c in columns if str(c).lower() == 'title'), None)
    abstract_col = next((c for c in columns if str(c).lower() in ['abstract', 'abstract_en', 'description']), None)
    
    title = str(row[title_col]).strip() if title_col else ""
    abstract = str(row[abstract_col]).strip() if abstract_col else ""
    
    text = f"{title} {abstract}".strip()
    return re.sub(r"\s+", " ", text) if len(text) > 5 else "[empty_text]"

# ====================================================================
# 🚀 ANALİZ BAŞLIYOR
# ====================================================================
def generate_detailed_report():
    print(f"📂 Model Yükleniyor: {MODEL_PATH}")
    
    model = XLMRobertaMultiLabel(MODEL_NAME, NUM_LABELS)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ HATA: '{MODEL_PATH}' bulunamadı!")
        return

    try:
        # strict=True yapıyoruz çünkü artık her şeyin birebir uyması lazım
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
        print("✅ Model Başarıyla ve Tam Uyumla Yüklendi!")
    except Exception as e:
        print(f"❌ MODEL YÜKLEME HATASI: {e}")
        return

    model.to(DEVICE)
    model.eval()

    print("📊 Veri Hazırlanıyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df_val = read_table(VAL_FILE)
    cols = df_val.columns.tolist()
    
    texts = [smart_combine_text(row, cols) for _, row in df_val.iterrows()]
    
    if texts[0] == "[empty_text]":
        print("🚨 UYARI: Metinler boş görünüyor!")
    
    labels = df_val[SDG_LABELS].values.astype(float)
    
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.FloatTensor(labels))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print("🔮 Tahminler Alınıyor...")
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            ids, mask, lbls = [b.to(DEVICE) for b in batch]
            logits = model(ids, mask)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(lbls.cpu().numpy())
            
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    
    preds = (all_probs > FINAL_THRESHOLD).astype(int)
    
    print(f"\n🏆 DETAYLI PERFORMANS KARNESİ (Eşik: {FINAL_THRESHOLD})")
    print("="*80)
    print(classification_report(all_targets, preds, target_names=FULL_SDG_LABELS, zero_division=0))
    
    # Excel'e Kaydet
    report_dict = classification_report(all_targets, preds, target_names=FULL_SDG_LABELS, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_excel("SDG_DETAYLI_ANALIZ_FINAL.xlsx")
    print(f"✅ Rapor Kaydedildi: SDG_DETAYLI_ANALIZ_FINAL.xlsx")

if __name__ == "__main__":
    generate_detailed_report()