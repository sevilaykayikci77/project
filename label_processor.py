import pandas as pd
import numpy as np

def process_labels_v2():
    # 1. Dosyaları Oku
    try:
        sep_df = pd.read_csv("SEP-28k_labels.csv")
    except FileNotFoundError:
        print("[HATA] SEP-28k_labels.csv bulunamadı!")
        return

    try:
        flu_df = pd.read_csv("FluencyBank_labels.csv")
    except FileNotFoundError:
        print("[HATA] FluencyBank_labels.csv bulunamadı!")
        return

    # --- FluencyBank Sütun İsimlerini Standartlaştır ---
    # Küçük harfli sütunları büyük harfe çeviriyoruz
    flu_df = flu_df.rename(columns={
        "show": "Show",
        "ep_id": "EpId",
        "clip_id": "ClipId",
        "start": "Start",
        "stop": "Stop"
    })

    # EPID padding (001, 002 formatı için kritik)
    flu_df["EpId"] = flu_df["EpId"].apply(lambda x: str(x).split('.')[0].zfill(3) if pd.notna(x) else "000")

    # --- BİRLEŞTİR ---
    # İki veri seti de artık aynı sütun yapısında olduğu için doğrudan alt alta ekliyoruz
    df = pd.concat([sep_df, flu_df], ignore_index=True)
    
    # Kekemelik türü sütunları
    stutter_cols = ['Prolongation', 'Block', 'SoundRep', 'WordRep']

    # 2. HESAPLAMALAR (Her iki veri seti için de ortak)
    
    # Toplam kekemelik oyu (Satır bazlı)
    df['total_stutter_votes'] = df[stutter_cols].sum(axis=1)
    
    # Toplam rater sayısı tahmini
    # (Hiç kekemelik yok diyenler + en az bir kekemelik türü işaretleyenlerin toplamı)
    df['calculated_raters'] = df['NoStutteredWords'] + df['total_stutter_votes']
    
    # Oransal Eşikleme (Ratio)
    df['stutter_ratio'] = df['total_stutter_votes'] / df['calculated_raters']
    
    # 3. VERİ KALİTESİ FİLTRESİ
    # Hem SEP-28k hem de FluencyBank için toplam rater sayısı 3 veya daha fazla olanları tutuyoruz
    df_clean = df[df['calculated_raters'] >= 3].copy()
    
    # 4. NİHAİ ETİKETLEME (Decision)
    # Eğer rater'ların %50 veya daha fazlası en az bir kekemelik işaretlediyse 1, değilse 0
    df_clean['is_stutter'] = (df_clean['stutter_ratio'] >= 0.5).astype(int)

    # --- İSTATİSTİKSEL RAPOR ---
    total_raw = len(df)
    total_filtered = len(df_clean)
    stutter_count = df_clean['is_stutter'].sum()
    fluent_count = total_filtered - stutter_count

    print(f"\n" + "="*45)
    print(f" NİHAİ ETİKETLEME VE KALİTE RAPORU (ORTAK MANTIK) ")
    print(f"="*45)
    print(f"Ham Veri Sayısı          : {total_raw}")
    print(f"Filtrelenmiş (Rater>=3)  : {total_filtered} (Eleme: {total_raw - total_filtered})")
    print(f"Nihai Kekeme (Label 1)   : {stutter_count} (%{round(stutter_count/total_filtered*100, 1) if total_filtered > 0 else 0})")
    print(f"Nihai Akıcı  (Label 0)   : {fluent_count} (%{round(fluent_count/total_filtered*100, 1) if total_filtered > 0 else 0})")
    print("-" * 45)

    # 5. KAYIT
    output_cols = ['Show', 'EpId', 'ClipId', 'Start', 'Stop', 'is_stutter', 'stutter_ratio', 'calculated_raters']
    df_clean[output_cols].to_csv('processed_labels.csv', index=False)
    print(f"[BAŞARILI] Tüm veriler ortak mantıkla işlendi. Toplam: {len(df_clean)} klip kaydedildi.")

if __name__ == "__main__":
    process_labels_v2()