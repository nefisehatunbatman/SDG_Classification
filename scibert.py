import argparse 
import pandas as pd
import re
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW 
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, hamming_loss
import numpy as np 
from tqdm import tqdm

# ====================================================================
# 1. KONFİGÜRASYON VE SABİTLER
# ====================================================================

# 🔥 KRİTİK DEĞİŞİKLİK: SciBERT Modeli Seçildi
# Bu model bilimsel makaleler (semanticscholar) üzerinde eğitilmiştir.
MODEL_NAME = "allenai/scibert_scivocab_uncased"

MAX_LEN = 256
SEED = 42 

# Etiket sütunları (SDG1, SDG2...)
SDG_LABELS = [
    f"SDG{i}" for i in range(1, 18)
] 

# Çıktı için tam SDG isimleri
FULL_SDG_LABELS = [
    "SDG 1: No Poverty", "SDG 2: Zero Hunger", "SDG 3: Good Health and Well-being",
    "SDG 4: Quality Education", "SDG 5: Gender Equality", "SDG 6: Clean Water and Sanitation",
    "SDG 7: Affordable and Clean Energy", "SDG 8: Decent Work and Economic Growth",
    "SDG 9: Industry, Innovation and Infrastructure", "SDG 10: Reduced Inequalities",
    "SDG 11: Sustainable Cities and Communities", "SDG 12: Responsible Consumption and Production",
    "SDG 13: Climate Action", "SDG 14: Life Below Water", "SDG 15: Life on Land",
    "SDG 16: Peace, Justice and Strong Institutions", "SDG 17: Partnerships for the Goals",
]

NUM_LABELS = len(SDG_LABELS)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan Cihaz: {DEVICE}")

def set_seed(seed_value):
    """Tekrarlanabilirliği sağlar."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
set_seed(SEED) 

# ====================================================================
# 2. VERİ OKUMA VE HAZIRLIK FONKSİYONLARI
# ====================================================================

def read_table(path: str, sheet: int = 0) -> pd.DataFrame:
    """Excel veya CSV dosyasını okur. Excel'de sadece ilk sayfayı okur."""
    print(f"Veri okunuyor: {path}")
    if path.endswith('.xlsx') or path.endswith('.xls'):
        return pd.read_excel(path, engine="openpyxl", sheet_name=sheet) 
    return pd.read_csv(path)

def combine_text(row):
    """SADECE İNGİLİZCE metinleri birleştirir ve KÜÇÜK HARFE çevirir (SciBERT uncased için)."""
    
    title_en = str(row.get("title", "")).strip() 
    descp_en = str(row.get("abstract", "")).strip()
    
    final_text = f"{title_en} {descp_en}".strip() 
    
    # 🔥 SciBERT 'uncased' olduğu için her şeyi küçük harfe çeviriyoruz
    final_text = re.sub(r"\s+", " ", final_text).lower()
    
    return final_text if final_text else "[empty_text]"

# ====================================================================
# 3. DATASET SINIFI
# ====================================================================
class SDGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer 
        self.max_length = max_length 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer( 
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

# ====================================================================
# 4. MODEL MİMARİSİ
# ====================================================================
class XLMRobertaMultiLabel(nn.Module):
    """
    SciBERT + Classification Head mimarisi. 
    (Sınıf adı XLM kalsa da içerik SciBERT'e göre çalışır, çünkü AutoModel kullanıyoruz)
    """
    def __init__(self, model_name, num_labels, dropout=0.3):
        super().__init__()
        self.xlm_roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.2),
            nn.Linear(384, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        if pooled_output is None:
             pooled_output = outputs.last_hidden_state[:, 0, :]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# ====================================================================
# 5. EĞİTİM VE DEĞERLENDİRME FONKSİYONLARI
# ====================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss() 
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return {
        'loss': total_loss / len(dataloader),
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'hamming_loss': hamming,
        'f1_per_class': f1_per_class
    }


def train_model(train_loader, val_loader, model, device, epochs=5, lr=2e-5, model_path="best_model.pt"):
    
    optimizer = AdamW([
        {'params': model.xlm_roberta.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr * 5} 
    ])

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = -1.0
    
    for epoch in range(epochs):
        print(f"\n{'='*50}\nEpoch {epoch + 1}/{epochs}\n{'='*50}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, F1 Micro: {val_metrics['f1_micro']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}")
        
        # Hedef Bazlı F1 Skorları Çıktısı
        print("\n--- SDG Hedef Bazlı F1 Skorları (%) ---")
        for i, sdg_label in enumerate(FULL_SDG_LABELS):
            score = val_metrics['f1_per_class'][i] * 100
            print(f"{sdg_label}: {score:.2f}%")
        print("----------------------------------------\n")

        if val_metrics['f1_micro'] > best_f1:
            best_f1 = val_metrics['f1_micro']
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved! F1: {best_f1:.4f} -> {model_path}")

    return model

# ====================================================================
# 6. TAHMİN (PREDICTION) FONKSİYONU
# ====================================================================
def predict_dataframe(df: pd.DataFrame, model, tokenizer, device, batch_size=16, threshold=0.5):
    model.eval()
    texts = [combine_text(row) for _, row in df.iterrows()]

    dataset = SDGDataset(texts, [[0]*NUM_LABELS]*len(texts), tokenizer) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    
    print("Tahminler yapılıyor...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    
    results = []
    for i in range(len(df)):
        row_data = {}
        probs_i = all_probs[i]
        
        for j, full_label in enumerate(FULL_SDG_LABELS):
             row_data[f"{full_label} (%)"] = round(probs_i[j] * 100, 2)

        top3_indices = np.argsort(probs_i)[::-1][:3] 
        for k, idx in enumerate(top3_indices, 1):
            row_data[f"Top{k} SDG"] = FULL_SDG_LABELS[idx]
            row_data[f"Top{k} (%)"] = round(probs_i[idx] * 100, 2)

        active_labels = [FULL_SDG_LABELS[j] for j, p in enumerate(probs_i) if p > threshold]
        row_data["Active SDGs"] = ", ".join(active_labels) if active_labels else "None"

        row_data["Text (preview)"] = texts[i][:200] + "..."
        results.append(row_data)

    return pd.DataFrame(results)

# ====================================================================
# 7. MAIN FONKSİYONU
# ====================================================================

def run_scibert_classifier_full(
    mode: str, 
    input_file: str, 
    output_file: str, 
    model_path: str, 
    epochs: int, 
    batch_size: int, 
    lr: float, 
    threshold: float
):
    # SciBERT Tokenizer Yükleniyor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = XLMRobertaMultiLabel(MODEL_NAME, NUM_LABELS).to(DEVICE)

    if mode == "train":
        print("\n=== EĞİTİM MODU (SciBERT + Tüm Veri) ===")
        
        df_full = read_table(input_file, sheet=0)
        
        texts = [combine_text(row) for _, row in df_full.iterrows()]
        labels = df_full[SDG_LABELS].values.astype(np.int32)
        
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=SEED
        )
        
        print(f"Toplam Veri: {len(df_full)}. Train: {len(X_train)}, Val: {len(X_val)}")
        
        train_dataset = SDGDataset(X_train, y_train, tokenizer)
        val_dataset = SDGDataset(X_val, y_val, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        train_model(train_loader, val_loader, model, DEVICE, epochs, lr, model_path)

    else: # mode == "predict"
        print("\n=== TAHMİN MODU (SciBERT) ===")
        
        try:
            print(f"Model yükleniyor: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except FileNotFoundError:
            raise FileNotFoundError(f"Tahmin için model dosyası bulunamadı: {model_path}. Önce eğitimi tamamlayın.")
        
        df = read_table(input_file, sheet=0)
        
        results_df = predict_dataframe(df, model, tokenizer, DEVICE, batch_size, threshold)

        merged = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            merged.to_excel(writer, index=False, sheet_name="SDG_Predictions")

        print(f"\n✅ Tamamlandı! Çıktı dosyası: {output_file}")


# ====================================================================
# ANA ÇALIŞTIRMA KISMI
# ====================================================================

if __name__ == "__main__":
    
    CALISMA_MODU = "train" 
    
    # GİRDİ DOSYASI (TÜM VERİ)
    GIRDI_DOSYASI = "/kaggle/input/tf-sdg/TF_SDG.xlsx" 
    
    # 🔥 SCIBERT İÇİN ÖZEL DOSYA İSİMLERİ
    CIKTI_DOSYASI = "tum_veri_SCIBERT_tahmin.xlsx"
    MODEL_YOLU = "tum_veri_best_model_SCIBERT.pt" 

    # Hiperparametreler
    EPOCH_SAYISI = 5
    BATCH_SIZE = 16
    LR = 2e-5
    THRESHOLD = 0.5 
    
    print(f"Seçilen Mod: {CALISMA_MODU.upper()} (MODEL: {MODEL_NAME})")
    
    try:
        run_scibert_classifier_full(
            mode=CALISMA_MODU,
            input_file=GIRDI_DOSYASI, 
            output_file=CIKTI_DOSYASI,
            model_path=MODEL_YOLU,
            epochs=EPOCH_SAYISI,
            batch_size=BATCH_SIZE,
            lr=LR,
            threshold=THRESHOLD
        )
    except Exception as e:
        print(f"\n🚨 Kritik Hata Oluştu: {e}")