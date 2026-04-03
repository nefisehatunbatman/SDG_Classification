import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="SDG SciBERT Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TASARIM ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    h1 { color: #2c3e50; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- GLOBAL AYARLAR ---
BASE_MODEL_NAME = "allenai/scibert_scivocab_uncased"
MY_MODEL_PATH = "tum_veri_best_model_SCIBERT.pt"
NUM_LABELS = 17 

SDG_LABELS = [
    "SDG 1: Yoksulluğa Son", "SDG 2: Açlığa Son", "SDG 3: Sağlık ve Kaliteli Yaşam",
    "SDG 4: Nitelikli Eğitim", "SDG 5: Cinsiyet Eşitliği", "SDG 6: Temiz Su ve Sanitasyon",
    "SDG 7: Erişilebilir ve Temiz Enerji", "SDG 8: İnsana Yakışır İş ve Büyüme",
    "SDG 9: Sanayi, Yenilikçilik ve Altyapı", "SDG 10: Eşitsizliklerin Azaltılması",
    "SDG 11: Sürdürülebilir Şehirler", "SDG 12: Sorumlu Üretim ve Tüketim",
    "SDG 13: İklim Eylemi", "SDG 14: Sudaki Yaşam", "SDG 15: Karasal Yaşam",
    "SDG 16: Barış ve Adalet", "SDG 17: Amaçlar için Ortaklıklar"
]

# --- MODEL YÜKLEME FONKSİYONU ---
@st.cache_resource
def load_model():
    try:
        # 1. Tokenizer Yükle
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
        # 2. Model İskeleti
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, 
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True
        )
        
        # 3. Ağırlıkları Yükle (GÜVENLİK AYARI DÜZELTİLDİ: weights_only=False)
        state_dict = torch.load(
            MY_MODEL_PATH, 
            map_location=torch.device('cpu'), 
            weights_only=False 
        )
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

# --- TAHMİN FONKSİYONU ---
# Hata almamak için tokenizer ve model'i argüman olarak alıyoruz
def predict_text(text, model_obj, tokenizer_obj):
    inputs = tokenizer_obj(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_obj(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.flatten().tolist()
    
    results = {SDG_LABELS[i]: probs[i] for i in range(len(SDG_LABELS)) if i < len(probs)}
    top_label = max(results, key=results.get)
    return top_label, results

# --- UYGULAMA AKIŞI ---

# 1. Modeli Yükle (Bu satır çok önemli, silinmemeli)
with st.spinner('SciBERT Modeli Yükleniyor...'):
    model, tokenizer, error_msg = load_model()

# 2. Yükleme Hatası Kontrolü
if error_msg:
    st.error(f"🚨 Model Yüklenemedi! Hata: {error_msg}")
    st.warning("Lütfen 'tum_veri_best_model_SCIBERT.pt' dosyasının doğru yerde olduğundan emin olun.")
    st.stop()

# 3. Arayüz
st.sidebar.image("https://sdgs.un.org/sites/default/files/2020-09/SDG_Wheel_Transparent_WEB.png", width=120)
st.sidebar.title("SDG Analiz Paneli")
menu = st.sidebar.radio("Menü", ["Canlı Analiz", "Dosya Yükle (Batch)"])

if menu == "Canlı Analiz":
    st.title("🌍 Akademik SDG Sınıflandırıcı (SciBERT)")
    st.markdown("Proje veya makale özetini girerek hangi Sürdürülebilir Kalkınma Amacı ile ilgili olduğunu analiz edin.")
    
    text_input = st.text_area("Metin Girişi:", height=150, placeholder="Metni buraya yapıştırın...")
    
    if st.button("Analiz Et"):
        if not text_input:
            st.warning("Lütfen bir metin girin.")
        else:
            # Fonksiyonu çağırırken model ve tokenizer'ı gönderiyoruz
            top_result, all_probs = predict_text(text_input, model, tokenizer)
            
            st.divider()
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.success(f"**Sonuç:**\n# {top_result}")
                confidence = all_probs[top_result]
                st.metric("Güven Skoru", f"%{confidence*100:.2f}")
                
            with c2:
                filtered_probs = {k: v for k, v in all_probs.items() if v > 0.01}
                df_chart = pd.DataFrame(list(filtered_probs.items()), columns=['SDG', 'Olasılık'])
                df_chart = df_chart.sort_values('Olasılık', ascending=True)
                
                fig = px.bar(df_chart, x='Olasılık', y='SDG', orientation='h', title="Olasılık Dağılımı", color='Olasılık')
                st.plotly_chart(fig, use_container_width=True)

elif menu == "Dosya Yükle (Batch)":
    st.header("📂 Çoklu Veri Analizi")
    uploaded_file = st.file_uploader("CSV Dosyası Seçin", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Yüklenen Veri:", df.head())
        
        if st.button("Tümünü Analiz Et"):
            results = []
            bar = st.progress(0)
            
            for i, row in df.iterrows():
                text = str(row[0])
                # Burada da model ve tokenizer'ı gönderiyoruz
                pred, _ = predict_text(text, model, tokenizer)
                results.append(pred)
                bar.progress((i + 1) / len(df))
            
            df['Tahmin'] = results
            st.success("İşlem Tamamlandı!")
            st.write(df.head())
            
            fig_pie = px.pie(df, names='Tahmin', title='Veri Seti SDG Dağılımı')
            st.plotly_chart(fig_pie)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Sonuçları İndir", csv, "sonuclar.csv", "text/csv")