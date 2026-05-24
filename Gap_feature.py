"""
HuBERT GAP Embedding Extraction (Sızıntısız - Üçlü Bölüm)
--------------------------------------------------------
Bu kod fine-tune edilmiş HuBERT modelinden Train, Val ve Test setleri için
AYRI AYRI 768 boyutlu GAP öznitelikleri çıkarır ve tek bir .npz dosyasına kaydeder.
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# =====================================================
# AYARLAR
# =====================================================
BASE_DIR = r"C:\Users\BIL MUH\Desktop\Damla\project_stutter\archive"
AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")
MODEL_PATH = os.path.join(BASE_DIR, "best_hubert_stuttering_model")

# Ablasyon kodunun okuyacağı features klasörünü ayarla
FEATURES_DIR = os.path.join(BASE_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

# Giriş CSV Dosyaları (Üçlü Yapı)
CSV_FILES = {
    "train": os.path.join(BASE_DIR, "train_split.csv"),
    "val":   os.path.join(BASE_DIR, "val_split.csv"),
    "test":  os.path.join(BASE_DIR, "test_split.csv")
}

MAX_LENGTH = 16000 * 3   # 3 saniye
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# MODEL VE FEATURE EXTRACTOR YÜKLEME
# =====================================================
print("Fine-tuned HuBERT modeli öznitelik çıkarımı için yükleniyor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = HubertModel.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

print(f"Kullanılan cihaz: {DEVICE}")

# =====================================================
# ÖZNİTELİK ÇIKARMA FONKSİYONU
# =====================================================
def extract_split_features(csv_path, split_name):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    
    print(f"\n🚀 [{split_name.upper()}] seti için öznitelik çıkarımı başladı... (Toplam: {len(df)} satır)")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        show = str(row["Show"]).strip()
        ep_id = str(row["EpId"]).strip()
        clip_id = str(row["ClipId"]).strip()

        if "fluencybank" in show.lower():
            ep_id = ep_id.zfill(3)

        filename = f"{show}_{ep_id}_{clip_id}.wav"
        full_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(full_path):
            continue

        # Ses dosyasını oku
        try:
            speech, sr = sf.read(full_path)
            if len(speech.shape) > 1:
                speech = speech[:, 0] # Mono yap
            if len(speech) < 400:
                continue
        except Exception: # Güvenli hata yakalama
            continue

        # Model girdisini hazırla
        inputs = feature_extractor(
            speech, sampling_rate=16000, return_tensors="pt",
            padding="max_length", truncation=True, max_length=MAX_LENGTH,
            return_attention_mask=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # GAP Embedding Hesapla
        with torch.no_grad():
            outputs = model(input_values=inputs["input_values"], attention_mask=inputs["attention_mask"])
            hidden_states = outputs.last_hidden_state # Boyut: [1, Sequence_Length, 768]
            
            # GÜVENLİ VE AKADEMİK DOĞRU GAP (Mean Pooling) ADIMI:
            embedding = torch.mean(hidden_states, dim=1)[0].cpu().numpy()

        X.append(embedding)
        y.append(int(row["is_stutter"]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"✅ {split_name.upper()} tamamlandı -> X shape: {X.shape}, y shape: {y.shape}")
    
    # Artık direkt kaydetmiyoruz, numpy array'leri geri döndürüyoruz
    return X, y

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    print("\n==============================")
    print("FEATURE EXTRACTION BAŞLADI")
    print("==============================")

    # Tüm setleri toplayacağımız sözlük
    all_features = {}

    for split_name, csv_path in CSV_FILES.items():

        print(f"\nİşleniyor: {split_name}")
        print(f"CSV PATH: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"[HATA] CSV bulunamadı: {csv_path}")
            continue

        # Arrayleri değişkene al
        X_data, y_data = extract_split_features(csv_path, split_name)
        
        # Sözlüğe npz key'lerine uygun formatta ekle
        all_features[f"X_{split_name}"] = X_data
        all_features[f"y_{split_name}"] = y_data

    # =====================================================
    # TEK BİR NPZ DOSYASINA KAYDETME ADIMI
    # =====================================================
    save_path = os.path.join(FEATURES_DIR, "hubert_gap_features.npz")
    
    # savez_compressed ile yerden tasarruf ederek tek dosyada topluyoruz
    np.savez_compressed(save_path, **all_features)
    
    print("\n==============================")
    print(f"✅ TÜM İŞLEMLER TAMAMLANDI.")
    print(f"📁 Dosya şuraya kaydedildi: {save_path}")
    print("==============================")