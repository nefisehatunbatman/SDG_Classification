# SDG Classification with Transformer Models

## Proje Amacı

Bu proje, akademik makalelerin başlık ve özetlerini analiz ederek, Birleşmiş Milletler'in 17 Sürdürülebilir Kalkınma Hedefi (SDG) ile olan ilişkisini otomatik olarak tespit etmeyi amaçlamaktadır.

Manuel sınıflandırmanın zaman alıcı ve hataya açık olması nedeniyle, bu çalışmada yapay zekâ tabanlı çok etiketli (multi-label) sınıflandırma yaklaşımı kullanılmıştır.

---

## Kullanılan Teknolojiler

- Python
- PyTorch
- Transformers (HuggingFace)
- NLP (Doğal Dil İşleme)
- SciBERT
- XLM-RoBERTa

---

## Model Mimarisi

Projede Transformer tabanlı modeller kullanılmıştır:

- XLM-RoBERTa
- SciBERT

Model, her SDG için bağımsız olasılık üreten **Sigmoid aktivasyon fonksiyonu** ile çalışır ve çok etiketli sınıflandırma yapar.

---

## Problem Türü

- Multi-label classification
- 17 farklı SDG etiketi
- Her makale birden fazla etikete sahip olabilir

---

## Veri Ön İşleme

- Tokenization (subword)
- Noise removal
- Text normalization

---

## Performans

- F1 Macro: ~0.65+
- F1 Micro: ~0.65+
- Top-1 Accuracy: ~0.77
- Top-3 Recall: ~0.91

---

## Deneyler

Farklı veri oranları ve model mimarileri test edilmiştir:

- XLM-RoBERTa (baseline)
- SciBERT (best model)

SciBERT modeli özellikle bilimsel metinlerde daha başarılı sonuçlar vermiştir.

---

## Proje Yapısı

```
project/
│── app.py
│── predict.py
│── model/
│── data/
│── results/
```

---

## Nasıl Çalıştırılır?

```bash
pip install -r requirements.txt
python app.py
```

---

## Öğrendiklerim

- Transformer mimarisi
- Multi-label classification
- NLP pipeline
- Model evaluation (F1, Recall vs.)
- Data imbalance handling

---

## Gelecek Çalışmalar

- API haline getirme
- Real-time inference
- Web arayüzü

---

## Geliştirici

Nefise Hatun Batman
