#asıl kod
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import os

# ====================================================================
# 1. AYARLAR (KONFİGÜRASYON)
# ====================================================================
class Config:
    MODEL_NAME = "allenai/scibert_scivocab_uncased"
    FILE_PATH = "/kaggle/input/tum-veri/TF_SDG.xlsx"  # Dosya yolunu kontrol et!
    OUTPUT_MODEL_PATH = "scibert_final_model.pt"
    OUTPUT_THRESHOLDS = "scibert_thresholds.npy"
    
    # Makalenin tamamını okuması için 512 yaptık
    MAX_LEN = 512 
    
    # 512 uzunluk GPU'yu zorlar, bu yüzden Batch Size'ı 8'e çektik (Güvenli Mod)
    BATCH_SIZE = 8 
    EPOCHS = 6
    LEARNING_RATE = 2e-5
    SEED = 42
    
    SDG_LABELS = [f"SDG{i}" for i in range(1, 18)]
    NUM_LABELS = len(SDG_LABELS)

# Cihaz Seçimi
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Çalışma Ortamı: {DEVICE}")

# Rastgeleliği Sabitle (Tekrarlanabilirlik İçin)
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(Config.SEED)

# ====================================================================
# 2. GELİŞMİŞ METRİK HESAPLAMA 
# ====================================================================
def compute_detailed_metrics(y_true, y_probs, threshold=0.5):

    # 1. Standart Tahminler (Eşik değerine göre 0 veya 1)
    y_pred = (y_probs > threshold).astype(int)
    
    # --- Klasik Metrikler ---
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    subset_acc = accuracy_score(y_true, y_pred) # Tam eşleşme (Zor metrik)
    

    top1_hits = 0
    top3_hits = 0
    total = len(y_true)
    
    for i in range(total):
        # Gerçekte var olan etiketlerin indeksleri
        true_indices = np.where(y_true[i] == 1)[0]
        if len(true_indices) == 0: continue # Etiketsiz veri varsa atla
        
        # Modelin tahmin ettiği olasılıkları sırala (Büyükten küçüğe)
        sorted_indices = np.argsort(y_probs[i])[::-1]
        
        # Top-1: En yüksek olasılıklı tahmin doğru mu?
        if sorted_indices[0] in true_indices:
            top1_hits += 1
            
        # Top-3: İlk 3 tahminin içinde en az 1 tane doğru var mı?
        # (Kullanıcıya 3 öneri sunsak işine yarar mı?)
        if any(idx in true_indices for idx in sorted_indices[:3]):
            top3_hits += 1
            
    top1_acc = top1_hits / total
    top3_acc = top3_hits / total
    
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "subset_acc": subset_acc,
        "top1_acc": top1_acc, # Raporun Yıldızı
        "top3_acc": top3_acc  # Yedek Yıldız
    }

# ====================================================================
# 3. VERİ HAZIRLIĞI
# ====================================================================
def prepare_data():
    print(f"📂 Veri Okunuyor: {Config.FILE_PATH}")
    if Config.FILE_PATH.endswith('.xlsx'):
        df = pd.read_excel(Config.FILE_PATH, engine='openpyxl')
    else:
        df = pd.read_csv(Config.FILE_PATH)
        
    # Metin Birleştirme (Başlık + Özet)
    texts = []
    for _, row in df.iterrows():
        t = str(row.get('title', '')).strip()
        a = str(row.get('abstract', '')).strip()
        texts.append(f"{t} {a}".lower()) # SciBERT uncased olduğu için lower()
        
    labels = df[Config.SDG_LABELS].values.astype(float)
    
    return train_test_split(texts, labels, test_size=0.15, random_state=Config.SEED)

class SDGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=Config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ====================================================================
# 4. MODEL MİMARİSİ
# ====================================================================
class SciBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        # 768 (SciBERT boyutu) -> 17 (SDG sayısı)
        self.classifier = nn.Linear(768, Config.NUM_LABELS)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token'ı alıyoruz (Cümlenin genel temsili)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled_output))

# ====================================================================
# 5. EĞİTİM DÖNGÜSÜ
# ====================================================================
def train_engine():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    train_texts, val_texts, train_labels, val_labels = prepare_data()
    
    train_dataset = SDGDataset(train_texts, train_labels, tokenizer)
    val_dataset = SDGDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    model = SciBERTClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() # Multi-label standart kayıp fonksiyonu
    
    # Scheduler: Öğrenme hızını yavaşça düşürür (Daha hassas öğrenme için)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    best_score = 0
    
    print("\n🚀 EĞİTİM BAŞLIYOR...")
    for epoch in range(Config.EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
            optimizer.zero_grad()
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            
            preds = model(ids, mask)
            loss = criterion(preds, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Patlayan gradyanları önle
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        # --- EVALUATION ---
        model.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                y = batch['labels'].to(DEVICE)
                
                preds = model(ids, mask)
                # Sigmoid ile olasılığa çevir (0.0 - 1.0 arası)
                probs = torch.sigmoid(preds).cpu().numpy()
                
                all_probs.extend(probs)
                all_targets.extend(y.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        # Metrikleri Hesapla 
        metrics = compute_detailed_metrics(all_targets, all_probs, threshold=0.5)
        
        print(f"\n📊 Epoch {epoch+1} Sonuçları:")
        print(f"   Train Loss  : {train_loss/len(train_loader):.4f}")
        print(f"   F1 Macro    : {metrics['f1_macro']:.4f}")
        print(f"   Top-1 Acc   : {metrics['top1_acc']:.4f} (En İyi Tahmin Doğruluğu)")
        print(f"   Top-3 Recall: {metrics['top3_acc']:.4f} (İlk 3'te Yakalama)")
        
        # En iyi modeli kaydet (Top-1 Accuracy'ye göre karar veriyoruz artık!)
        if metrics['top1_acc'] > best_score:
            best_score = metrics['top1_acc']
            torch.save(model.state_dict(), Config.OUTPUT_MODEL_PATH)
            
            # Dinamik Eşikleri Hesapla ve Kaydet
            best_thresholds = []
            for i in range(Config.NUM_LABELS):
                # Basit bir threshold optimizasyonu
                best_t = 0.5
                best_f1 = 0
                for t in np.arange(0.2, 0.8, 0.05):
                    p = (all_probs[:, i] > t).astype(int)
                    f = f1_score(all_targets[:, i], p)
                    if f > best_f1:
                        best_f1 = f
                        best_t = t
                best_thresholds.append(best_t)
            
            np.save(Config.OUTPUT_THRESHOLDS, np.array(best_thresholds))
            print(f"✅ Yeni En İyi Model Kaydedildi! (Top-1 Acc: {best_score:.4f})")
            
    print("\n🏁 Eğitim Tamamlandı.")

if __name__ == "__main__":
    train_engine()