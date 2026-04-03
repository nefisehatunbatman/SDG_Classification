import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from math import ceil

# ====================================================================
# A. KONFİGÜRASYON VE PARAMETRELER
# ====================================================================

GIRDI_DOSYASI = "TF_SDG.xlsx"
EGITIM_CIKTI_DOSYASI = "4egitim_dengelenmis_orneklem.xlsx"
DOGRULAMA_CIKTI_DOSYASI = "4dogrulama_orneklem.xlsx"
SEED = 42

# 🔥 YENİ PARAMETRE: Veri setinin kaçta kaçının örneklem olarak alınacağı.
SAMPLING_RATE = 0.50  # %10'unu al (1000 satırlık veriden 100 satır alınır)

# Oversample hedefi: Eğitim setinde her SDG'nin ulaşmasını istediğiniz minimum sayı.
OVERSAMPLE_HEDEF = 150 

# SDG etiket sütunları
SDG_LABELS = [
    f"SDG{i}" for i in range(1, 18)
]

# ====================================================================
# B. FONKSİYONLAR
# ====================================================================

def read_data(path: str) -> pd.DataFrame:
    """Excel dosyasını okur."""
    print(f"Veri okunuyor: {path}")
    try:
        return pd.read_excel(path, engine="openpyxl", sheet_name=0)
    except FileNotFoundError:
        print(f"HATA: {path} dosyası bulunamadı. Lütfen yükleyin.")
        exit()

def oversample_data(df_train: pd.DataFrame, min_target: int) -> pd.DataFrame:
    """Eğitim DataFrame'ini nadir sınıflar için çoğaltır."""
    
    y_train = df_train[SDG_LABELS].values
    counts = y_train.sum(axis=0) 
    
    df_resampled = df_train.copy()
    
    print("\n--- ÖRNEKLEM DENGELEME BAŞLADI (Hedef: %d) ---" % min_target)
    
    for i in range(y_train.shape[1]):
        current_count = counts[i]
        
        if current_count < min_target:
            
            minority_indices = df_train.index[df_train[SDG_LABELS[i]] == 1]
            multiplier = int(ceil(min_target / current_count))
            multiplier = min(multiplier, 5) # Maksimum 5 kat ile sınırla

            df_to_duplicate = df_train.loc[minority_indices]
            
            for _ in range(multiplier - 1):
                 df_resampled = pd.concat([df_resampled, df_to_duplicate], ignore_index=True)
            
            print(f"SDG {i+1} ({SDG_LABELS[i]}): Mevcut: {current_count:4d} -> Çarpan: {multiplier:2d} -> Yeni Tahmini Sayı: {current_count * multiplier}")

    print(f"--- ÖRNEKLEM DENGELEME BİTTİ (Toplam örnek: {len(df_resampled)}) ---")
    
    return df_resampled

# ====================================================================
# C. ANA ÇALIŞTIRMA KISMI
# ====================================================================

def main_arac_yeni():
    df_tam = read_data(GIRDI_DOSYASI)
    
    print(f"Orijinal Toplam Satır: {len(df_tam)}")

    # 1. 🔥 VERİ KÜÇÜLTME (Örneklem Alma)
    if SAMPLING_RATE < 1.0:
        # Rastgele örnekleme ile veriyi küçült.
        # Bu işlem, etiket sütunları dikkate alınmadan yapılır.
        df_orneklem = df_tam.sample(frac=SAMPLING_RATE, random_state=SEED)
        print(f"Örneklem Boyutu ({SAMPLING_RATE * 100:.0f}%): {len(df_orneklem)}")
    else:
        df_orneklem = df_tam

    # 2. Train/Val Split (Küçültülmüş örneklem üzerinde)
    # Stratify kullanılamaz, çünkü bu aşamada tüm etiketleri içeren tek bir sütun yok.
    df_train, df_val = train_test_split(
        df_orneklem, test_size=0.2, random_state=SEED
    )
    
    print(f"Küçültülmüş Örneklem Üzerinden Ayrım: Train: {len(df_train)}, Val: {len(df_val)}")

    # 3. Eğitim Setini Dengele (Oversampling)
    df_train_dengelenmis = oversample_data(df_train, OVERSAMPLE_HEDEF)

    # 4. Dosyaları Kaydet
    with pd.ExcelWriter(EGITIM_CIKTI_DOSYASI, engine="openpyxl") as writer:
        df_train_dengelenmis.to_excel(writer, index=False, sheet_name="Egitim")
    
    with pd.ExcelWriter(DOGRULAMA_CIKTI_DOSYASI, engine="openpyxl") as writer:
        df_val.to_excel(writer, index=False, sheet_name="Dogrulama")
        
    print(f"\n✅ Veri Hazırlığı Tamamlandı!")
    print(f"Eğitim Verisi Kaydedildi: {EGITIM_CIKTI_DOSYASI}")
    print(f"Doğrulama Verisi Kaydedildi: {DOGRULAMA_CIKTI_DOSYASI}")


if __name__ == "__main__":
    main_arac_yeni()