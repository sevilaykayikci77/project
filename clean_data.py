import os
import pandas as pd
import librosa
import numpy as np
import shutil  # Dosya kopyalamak için gerekli standart kütüphane
from tqdm import tqdm

# --- YOLLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "processed_labels.csv")
DELETED_CSV_PATH = os.path.join(BASE_DIR, "deleted_labels.csv")

# HAM SESLERİN BULUNDUĞU KAYNAK DİZİN
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "clips", "stuttering-clips", "clips")

# İŞLENMİŞ/TEMİZ SESLERİN GEÇECEĞİ HEDEF DİZİN
AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")

# Hedef klasör yoksa otomatik olarak oluşturulmasını sağlıyoruz
os.makedirs(AUDIO_DIR, exist_ok=True)


def clean_and_verify_data():
    print("="*55)
    print(" VERİ TEMİZLİĞİ, DOĞRULAMA VE AKTARIM OPERASYONU ")
    print("="*55)

    if not os.path.exists(CSV_PATH):
        print(f"[HATA] CSV dosyası bulunamadı: {CSV_PATH}")
        return

    if not os.path.exists(RAW_AUDIO_DIR):
        print(f"[HATA] Klasör bulunamadı: {RAW_AUDIO_DIR}")
        return

    # Klasördeki tüm dosyaları birebir listele ve küçük harfe çevir
    folder_files = os.listdir(RAW_AUDIO_DIR)
    folder_files_lower = [f.lower() for f in folder_files]
    
    print(f"Klasörde toplam {len(folder_files)} fiziksel dosya var.")
    if len(folder_files) > 0:
        print(f"Klasörden örnek dosya adı: '{folder_files[0]}'")
    print("-" * 55)

    df = pd.read_csv(CSV_PATH)
    initial_count = len(df)
    
    to_drop = []      
    drop_reasons = [] 
    error_sample_count = 0 # Ekrana basılacak hata örneği sayısı
    copied_count = 0       # Kaç dosya kopyalandığını saymak için

    print(f"Toplam {initial_count} CSV kaydı taranıyor ve temiz olanlar taşınıyor...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Verilerin etrafındaki olası gizli boşlukları temizle (.strip())
        show_name = str(row['Show']).strip()
        clip_id = str(row['ClipId']).strip()
        
        # EpId için hem sıfırlı (001) hem de sıfırsız (1) versiyonu güvenliğe alalım
        ep_id_raw = str(row['EpId']).split('.')[0].strip()
        ep_id_padded = ep_id_raw.zfill(3)
        
        # Olası iki dosya adı formatını da türetelim
        filename_padded = f"{show_name}_{ep_id_padded}_{clip_id}.wav".lower()
        filename_raw = f"{show_name}_{ep_id_raw}_{clip_id}.wav".lower()

        # Klasörde bu iki isimden biri var mı?
        actual_filename = None
        if filename_padded in folder_files_lower:
            actual_filename = folder_files[folder_files_lower.index(filename_padded)]
        elif filename_raw in folder_files_lower:
            actual_filename = folder_files[folder_files_lower.index(filename_raw)]

        # --- EĞER DOSYA BULUNAMAZSA DEDEKTİFİ DEVREYE SOK ---
        if actual_filename is None:
            to_drop.append(idx)
            drop_reasons.append("Dosya Bulunamadi")
            
            # İlk 5 hatanın detayını ekrana basıp sorunu teşhis edelim
            if error_sample_count < 5:
                print(f"\n\n[DEDEKTİF] Eşleşme Sağlanamadı (Satır {idx}):")
                print(f"  -> CSV'deki Veriler : Show={show_name} | EpId={ep_id_raw} | ClipId={clip_id}")
                print(f"  -> Aranan İsim (Sıfırlı)  : '{filename_padded}'")
                print(f"  -> Aranan İsim (Sıfırsız) : '{filename_raw}'")
                
                # Klasörde şov adına benzeyen bir dosya var mı diye bakalım
                matches = [f for f in folder_files if show_name.lower() in f.lower()]
                if matches:
                    print(f"  -> Klasördeki Benzer Dosya: '{matches[0]}' (Farkı görebiliyor musun?)")
                else:
                    print(f"  -> Klasörde '{show_name}' ismiyle başlayan HİÇBİR dosya yok!")
                print("-" * 40)
                error_sample_count += 1
            continue

        # Dosya bulunduysa yola devam et
        filepath = os.path.join(RAW_AUDIO_DIR, actual_filename)

        # 2. Dosya Boyutu 0 mı?
        if os.path.getsize(filepath) == 0:
            to_drop.append(idx)
            drop_reasons.append("Bos Dosya (0 Byte)")
            continue

        # 3. İçerik Analizi
        try:
            y, sr = librosa.load(filepath, sr=16000, duration=0.1)
            if len(y) < 160 or np.all(y == 0):
                to_drop.append(idx)
                drop_reasons.append("Sessiz veya Cok Kisa Ses")
                continue  # Hatalıysa kopyalama adımını atla
        except Exception as e:
            to_drop.append(idx)
            drop_reasons.append("Bozuk Dosya (Okunamadi)")
            continue  # Hatalıysa kopyalama adımını atla

        # -------------------------------------------------------------
        # 4. AKTARIM ADIMI: Dosya tüm testleri geçtiyse hedef klasöre kopyala
        # -------------------------------------------------------------
        dest_filepath = os.path.join(AUDIO_DIR, actual_filename)
        
        # Eğer dosya hedefte zaten yoksa kopyala (Gereksiz disk yazımını önler)
        if not os.path.exists(dest_filepath):
            shutil.copy2(filepath, dest_filepath)  # copy2 meta dataları da korur
            copied_count += 1

    # --- KAYIT İŞLEMLERİ ---
    if len(to_drop) > 0:
        df_deleted = df.iloc[to_drop].copy()
        df_deleted['Reason'] = drop_reasons
        
        if os.path.exists(DELETED_CSV_PATH):
            df_deleted.to_csv(DELETED_CSV_PATH, mode='a', header=False, index=False)
        else:
            df_deleted.to_csv(DELETED_CSV_PATH, index=False)

    df_clean = df.drop(to_drop).reset_index(drop=True)
    df_clean.to_csv(CSV_PATH, index=False)

    print("\n" + "="*55)
    print(f"TEMİZLİK VE TAŞIMA TAMAMLANDI")
    print(f"İlk Kayıt Sayısı      : {initial_count}")
    print(f"Silinen Kayıt Sayısı  : {len(to_drop)}")
    print(f"Yeni Taşınan Dosya    : {copied_count}")
    print(f"Kalan Temiz Toplam    : {len(df_clean)} (processed_clips içinde hazır)")
    print("="*55)

if __name__ == "__main__":
    clean_and_verify_data()