"""
ADIM 5: HuBERT ABLASYON ÇALIŞMASI — SİSTEMATİK ML TARAMA
===================================================
Giriş  : features/hubert_gap_features.npz
Çıktı  : results/ablasyon_sonuclari.csv
         results/ablasyon_ozet.txt
         figures/ablasyon/ablasyon_heatmap.png

KOMBİNASYONLAR:
  Özellik kaynağı (1): Hubert_GAP
  Özellik seçici (4):  HAM (Ham/Tümü), KBEST (SelectKBest), LASSO, TREE (Ağaç Tabanlı)
  Sınıflandırıcı (8):  SVM_Lin, SVM_RBF, SVM_Quad, SVM_Cub, RF, XGBoost, CatBoost
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# KONFİGÜRASYON
CONFIG = {
    "feature_dir"    : "features",
    "result_dir"     : "results",
    "fig_dir"        : "figures/ablasyon",
    "k_best"         : 500,    # SelectKBest ile seçilecek özellik sayısı
    "lasso_cv"       : 5,
    "thresh_start"   : 0.25,
    "thresh_end"     : 0.75,
    "thresh_step"    : 0.01,
    "random_state"   : 42,
}

os.makedirs(CONFIG["result_dir"], exist_ok=True)
os.makedirs(CONFIG["fig_dir"],    exist_ok=True)

# ÖZELLİK DOSYALARI HARİTASI
FEATURE_FILES = {
    'Hubert_GAP': "hubert_gap_features.npz",
}

# SINIFLANDIRICI TANIMLARI
def get_classifiers():
    clfs = {
        'SVM_Lin' : SVC(kernel='linear', C=1.0, probability=True, random_state=CONFIG["random_state"]),
        'SVM_RBF' : SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=CONFIG["random_state"]),
        'SVM_Quad': SVC(kernel='poly', degree=2, C=0.1, coef0=1, gamma='scale', probability=True, random_state=CONFIG["random_state"]),
        'SVM_Cub' : SVC(kernel='poly', degree=3, C=0.1, coef0=1, gamma='scale', probability=True, random_state=CONFIG["random_state"]),
        
        'RF'      : RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=CONFIG["random_state"], n_jobs=-1),
        
        'XGBoost' : XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=CONFIG["random_state"], eval_metric='logloss', verbosity=0),
        
        'CatBoost': CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, auto_class_weights='Balanced', random_seed=CONFIG["random_state"], verbose=0)
    }
    return clfs

# VERİ YÜKLEME
def load_features(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        'X_train'  : d['X_train'],
        'y_train'  : d['y_train'],
        'X_val'    : d['X_val'],
        'y_val'    : d['y_val'],
        'X_test'   : d['X_test'],
        'y_test'   : d['y_test'],
    }

# ÖZELLİK SEÇİCİLER (Feature Selection)
def apply_feature_selection(X_train, y_train, X_val, X_test, method):
    if method == 'ham':
        return X_train, X_val, X_test, X_train.shape[1], "Ham özellikler"

    elif method == 'kbest':
        mm = MinMaxScaler()
        X_tr_mm = mm.fit_transform(X_train)
        X_va_mm = mm.transform(X_val)
        X_te_mm = mm.transform(X_test)
        
        k = min(CONFIG["k_best"], X_tr_mm.shape[1])
        sel = SelectKBest(f_classif, k=k)
        X_tr_s = sel.fit_transform(X_tr_mm, y_train)
        X_va_s = sel.transform(X_va_mm)
        X_te_s = sel.transform(X_te_mm)
        return X_tr_s, X_va_s, X_te_s, k, f"SelectKBest top-{k}"

    elif method == 'lasso':
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_train)
        X_va_sc = sc.transform(X_val)
        X_te_sc = sc.transform(X_test)
        
        lasso = LassoCV(cv=CONFIG["lasso_cv"], max_iter=10000, random_state=CONFIG["random_state"], n_jobs=-1)
        sel = SelectFromModel(lasso)
        
        X_tr_s = sel.fit_transform(X_tr_sc, y_train)
        X_va_s = sel.transform(X_va_sc)
        X_te_s = sel.transform(X_te_sc)
        
        n_features = X_tr_s.shape[1]
        if n_features == 0:
            print("    ⚠ LASSO tüm özellikleri eledi → ham kullanılıyor")
            return X_tr_sc, X_va_sc, X_te_sc, X_tr_sc.shape[1], "LASSO(boş→ham)"
        return X_tr_s, X_va_s, X_te_s, n_features, f"LASSO n={n_features}"

    elif method == 'tree':
        rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG["random_state"], n_jobs=-1)
        sel = SelectFromModel(rf, threshold='median')
        
        X_tr_s = sel.fit_transform(X_train, y_train)
        X_va_s = sel.transform(X_val)
        X_te_s = sel.transform(X_test)
        
        n_features = X_tr_s.shape[1]
        return X_tr_s, X_va_s, X_te_s, n_features, f"Tree-based n={n_features}"

    else:
        raise ValueError(f"Bilinmeyen yöntem: {method}")

# METRİKLER VE EŞİK (THRESHOLD) OPTİMİZASYONU
def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sens, spec = 0.0, 0.0
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    return acc, sens, spec, auc

def optimize_threshold(y_true, probs):
    """Validation seti üzerinde accuracy'yi maksimize eden en iyi eşik değerini bulur."""
    best_val, best_t = 0.0, 0.5
    for t in np.arange(CONFIG["thresh_start"], CONFIG["thresh_end"], CONFIG["thresh_step"]):
        preds = (probs >= t).astype(int)
        val = accuracy_score(y_true, preds)
        if val > best_val:
            best_val, best_t = val, round(float(t), 2)
    return best_t, best_val


# ANA DÖNGÜ
print("ADIM 5: HuBERT ABLASYON ÇALIŞMASI BAŞLIYOR")

datasets = {}
for feat_name, fname in FEATURE_FILES.items():
    fpath = os.path.join(CONFIG["feature_dir"], fname)
    data = load_features(fpath)
    if data is not None:
        datasets[feat_name] = data
        print(f"  ✓ {feat_name:15s} | Train: {data['X_train'].shape} | Val: {data['X_val'].shape} | Test: {data['X_test'].shape}")
    else:
        print(f"  ⚠ {feat_name}: dosya bulunamadı ({fname})")

if not datasets:
    raise FileNotFoundError("Hiçbir özellik dosyası yüklenemedi. features/hubert_gap_features.npz kontrol edin.")

SEL_METHODS = ['ham', 'kbest', 'lasso', 'tree']
all_results = []

n_clf = len(get_classifiers())
total_combs = len(datasets) * len(SEL_METHODS) * n_clf
done = 0

print(f"\nToplam kombinasyon: {total_combs}")

for feat_name, data in datasets.items():
    X_tr = data['X_train']; y_tr = data['y_train']
    X_va = data['X_val'];   y_va = data['y_val']
    X_te = data['X_test'];  y_te = data['y_test']

    for sel_method in SEL_METHODS:
        print(f"\n── {feat_name} + {sel_method.upper()} ──")

        try:
            X_tr_s, X_va_s, X_te_s, n_feat, sel_info = apply_feature_selection(X_tr, y_tr, X_va, X_te, sel_method)
            print(f"  Seçilen özellik: {n_feat} ({sel_info})")
        except Exception as e:
            print(f"  ⚠ Özellik seçimi hatası: {e} — atlanıyor")
            continue

        # Modeller için standartlaştırma (Lasso içinde yapılmış olsa da modeller için tekrar yapıyoruz)
        sc_main = StandardScaler()
        X_tr_sc = sc_main.fit_transform(X_tr_s)
        X_va_sc = sc_main.transform(X_va_s)
        X_te_sc = sc_main.transform(X_te_s)

        for clf_name, clf in get_classifiers().items():
            done += 1
            try:
                # Modeli eğit
                clf.fit(X_tr_sc, y_tr)

                # Val ve Test için olasılıkları tahmin et
                probs_va = clf.predict_proba(X_va_sc)[:, 1]
                probs_te = clf.predict_proba(X_te_sc)[:, 1]

                # Validation üzerinde optimal eşiği (threshold) bul
                best_thresh, _ = optimize_threshold(y_va, probs_va)
                
                # Bulunan eşiği kullanarak tahmin üret ve metrikleri hesapla
                preds_va = (probs_va >= best_thresh).astype(int)
                val_acc, val_sens, val_spec, val_auc = compute_metrics(y_va, preds_va, probs_va)

                preds_te = (probs_te >= best_thresh).astype(int)
                te_acc, te_sens, te_spec, te_auc = compute_metrics(y_te, preds_te, probs_te)

                # Sonuçları listeye ekle
                row = {
                    'Özellik Kaynağı' : feat_name,
                    'Özellik Seçici'  : sel_method.upper(),
                    'Sınıflandırıcı'  : clf_name,
                    'n_özellik'       : n_feat,
                    'Optimal_Thresh'  : best_thresh,
                    'Val_Acc'         : round(val_acc, 4),
                    'Val_AUC'         : round(val_auc, 4),
                    'Test_Acc'        : round(te_acc, 4),
                    'Test_AUC'        : round(te_auc, 4),
                    'Test_Sens'       : round(te_sens, 4),
                    'Test_Spec'       : round(te_spec, 4),
                }
                all_results.append(row)

                print(f"  [{done:2d}/{total_combs}] {clf_name:10s} | Val_Acc: {val_acc:.4f} | Test_Acc: {te_acc:.4f} | Test_AUC: {te_auc:.4f}")

            except Exception as e:
                print(f"  [{done:2d}/{total_combs}] {clf_name:10s} | HATA: {e}")

# SONUÇLARI KAYDET
df = pd.DataFrame(all_results)
df = df.sort_values('Val_Acc', ascending=False).reset_index(drop=True)

csv_path = os.path.join(CONFIG["result_dir"], "ablasyon_sonuclari.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# ÖZET RAPOR
ozet_path = os.path.join(CONFIG["result_dir"], "ablasyon_ozet.txt")
with open(ozet_path, "w", encoding="utf-8") as f:
    f.write("HuBERT ABLASYON ÇALIŞMASI — ÖZET RAPOR\n")
    f.write("=" * 80 + "\n\n")
    
    header = f"{'Seçici':8s} {'Clf':10s} {'n_feat':7s} {'Thresh':7s} {'Val_Acc':8s} {'Val_AUC':8s} {'Test_Acc':8s} {'Test_AUC':8s}\n"
    f.write(header)
    f.write("-" * 80 + "\n")
    for _, row in df.head(30).iterrows():
        f.write(f"{row['Özellik Seçici']:8s} {row['Sınıflandırıcı']:10s} {int(row['n_özellik']):7d} {row['Optimal_Thresh']:7.2f} {row['Val_Acc']:8.4f} {row['Val_AUC']:8.4f} {row['Test_Acc']:8.4f} {row['Test_AUC']:8.4f}\n")

print(f"\n✓ Tüm sonuçlar ({csv_path}) ve özet rapor ({ozet_path}) kaydedildi.")

# ISI HARİTASI (Seçici vs Sınıflandırıcı)
try:
    pivot = df.pivot_table(values='Val_Acc', index='Özellik Seçici', columns='Sınıflandırıcı', aggfunc='max')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, linewidths=0.5, annot_kws={"size": 10})
    ax.set_title('Ablasyon — Özellik Seçici × Sınıflandırıcı (Validation Accuracy)', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    heatmap_path = os.path.join(CONFIG["fig_dir"], "ablasyon_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Isı haritası kaydedildi: {heatmap_path}")
except Exception as e:
    print(f"⚠ Isı haritası oluşturulamadı: {e}")