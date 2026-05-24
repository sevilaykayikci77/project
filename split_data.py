import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold  # Değişen kısım burası!
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "processed_labels.csv")
SEED = 42

def deterministic_group_split():
    print("="*55)
    print(" SIZINTISIZ VE STRATIFIED (DENGELİ) VERI BÖLME ")
    print("="*55)

    df = pd.read_csv(CSV_PATH)
    
    # 1. KİŞİ (SPEAKER) KİMLİĞİ OLUŞTURMA
    def get_speaker_id(row):
        show = str(row['Show']).strip()
        epid = str(row['EpId']).strip()
        return f"{show}_{epid}"
            
    df['Speaker_ID'] = df.apply(get_speaker_id, axis=1)
    
    groups = df['Speaker_ID'].values
    y = df['is_stutter'].values

    # 2. ÖNCE TEST SETİNİ AYIR (~%15)
    # StratifiedGroupKFold n_splits=7 yaparsak, 1 parça test olur (1/7 = ~%14.2) 
    # Veya n_splits=6 yaparsan 1/6 = ~%16.6 olur. n_splits=7 en ideal %15'e yakındır.
    sgkf_test = StratifiedGroupKFold(n_splits=7)
    
    # split() fonksiyonu jeneratör olduğu için ilk katmanı alıyoruz
    train_val_idx, test_idx = next(sgkf_test.split(df, y, groups))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    # 3. KALANI TRAIN VE VAL OLARAK BÖL (~%15 Val, kalanı Train)
    # Kalan veri (6 parça) içinden 1 parçayı Val yapacağız (1/6 = ~%16.6)
    groups_train_val = df_train_val['Speaker_ID'].values
    y_train_val = df_train_val['is_stutter'].values
    
    sgkf_val = StratifiedGroupKFold(n_splits=6)
    train_idx, val_idx = next(sgkf_val.split(df_train_val, y_train_val, groups_train_val))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # 4. MUTLAK SIZINTI KONTROLÜ (KANIT)
    train_spk = set(df_train['Speaker_ID'])
    val_spk = set(df_val['Speaker_ID'])
    test_spk = set(df_test['Speaker_ID'])
    
    print("\n─── Konuşmacı Sızıntısı Raporu ───")
    print(f"Train ∩ Val   : {len(train_spk & val_spk)} ortak kişi")
    print(f"Train ∩ Test  : {len(train_spk & test_spk)} ortak kişi")
    print(f"Val ∩ Test    : {len(val_spk & test_spk)} ortak kişi")
    
    if len(train_spk & val_spk) > 0 or len(train_spk & test_spk) > 0 or len(val_spk & test_spk) > 0:
        print("\n[KRİTİK HATA] İzolasyon başarısız oldu! Kodu durdurun.")
        return
    else:
        print("-> İzolasyon BAŞARILI. Sızıntı SIFIR.")
    
    # 5. DOSYALARI KAYDET (Yolları BASE_DIR ile güvenli hale getirdik)
    df_train.drop(columns=['Speaker_ID']).to_csv(os.path.join(BASE_DIR, "train_split.csv"), index=False)
    df_val.drop(columns=['Speaker_ID']).to_csv(os.path.join(BASE_DIR, "val_split.csv"), index=False)
    df_test.drop(columns=['Speaker_ID']).to_csv(os.path.join(BASE_DIR, "test_split.csv"), index=False)
    
    print("\n─── Dağılım Özeti ───")
    print(f"Train : {len(df_train):>5} klip | Kişi: {len(train_spk):>3} | Kekeme Oranı: %{round(df_train['is_stutter'].mean()*100, 1)}")
    print(f"Val   : {len(df_val):>5} klip | Kişi: {len(val_spk):>3} | Kekeme Oranı: %{round(df_val['is_stutter'].mean()*100, 1)}")
    print(f"Test  : {len(df_test):>5} klip | Kişi: {len(test_spk):>3} | Kekeme Oranı: %{round(df_test['is_stutter'].mean()*100, 1)}")
    print("="*55)

if __name__ == "__main__":
    deterministic_group_split()