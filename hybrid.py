# ==============================================================================
# STUTTERING DETECTION — IMPROVED ENSEMBLE (FN OPTIMIZED)
# XGBoost + SVM + RandomForest (NORMALISED CM INTEGRATED)
# ==============================================================================

import os
import warnings
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    recall_score,
    roc_curve 
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."

CFG = {
    
    "train_features_path": os.path.join(SCRIPT_DIR, "X_train_s_selected.npy"), 
    "train_labels_path": os.path.join(SCRIPT_DIR, "y_train_s_selected.npy"),

    "val_features_path": os.path.join(SCRIPT_DIR, "X_val_s_selected.npy"),
    "val_labels_path": os.path.join(SCRIPT_DIR, "y_val_s_selected.npy"),

    "test_features_path": os.path.join(SCRIPT_DIR, "X_test_s_selected.npy"),
    "test_labels_path": os.path.join(SCRIPT_DIR, "y_test_s_selected.npy"),

    "output_dir": os.path.join(SCRIPT_DIR, "ensemble_outputs"),

    "random_seed": 42,
    "n_folds": 5,
    "n_iterations": 3000,   

    "xgb_params": dict(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50 
    ),

    "svm_params": dict(
        kernel="rbf",
        C=1.2,
        probability=True,
        class_weight="balanced",
        cache_size=1000 
    ),

    "rf_params": dict(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
}

MODEL_NAMES = ["XGBoost", "SVM", "RandomForest"]
CLASS_NAMES = ["Akici(0)", "Kekeme(1)"]

os.makedirs(CFG["output_dir"], exist_ok=True)

# ==============================================================================
# LOAD DATA
# ==============================================================================

def load_data():
    X_train = np.load(CFG["train_features_path"]).astype(np.float32)
    y_train = np.load(CFG["train_labels_path"]).astype(np.int32)

    X_val = np.load(CFG["val_features_path"]).astype(np.float32)
    y_val = np.load(CFG["val_labels_path"]).astype(np.int32)

    X_test = np.load(CFG["test_features_path"]).astype(np.float32)
    y_test = np.load(CFG["test_labels_path"]).astype(np.int32)

    print("\n📊 DATA LOADED")
    print("Train:", X_train.shape, np.bincount(y_train))
    print("Val  :", X_val.shape, np.bincount(y_val))
    print("Test :", X_test.shape, np.bincount(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================================================
# OOF TRAINING & VAL/TEST INFERENCE
# ==============================================================================

def train_and_infer(X_train, y_train, X_val, y_val, X_test):
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["random_seed"])

    val_preds = [np.zeros(len(X_val)) for _ in MODEL_NAMES]
    test_preds = [np.zeros(len(X_test)) for _ in MODEL_NAMES]

    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\nFold {fold}")

        X_tr, X_va = X_train[tr], X_train[va]
        y_tr, y_va = y_train[tr], y_train[va]

        # Ölçekleme
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        scale_pos_weight = 1.0

        # 1. XGBOOST
        xgb = XGBClassifier(**CFG["xgb_params"], scale_pos_weight=scale_pos_weight)
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        val_preds[0] += xgb.predict_proba(X_val_scaled)[:, 1] / CFG["n_folds"]
        test_preds[0] += xgb.predict_proba(X_test_scaled)[:, 1] / CFG["n_folds"]

        # 2. SVM
        svm = SVC(**CFG["svm_params"])
        svm.fit(X_tr, y_tr)
        val_preds[1] += svm.predict_proba(X_val_scaled)[:, 1] / CFG["n_folds"]
        test_preds[1] += svm.predict_proba(X_test_scaled)[:, 1] / CFG["n_folds"]

        # 3. RF
        rf = RandomForestClassifier(**CFG["rf_params"])
        rf.fit(X_tr, y_tr)
        val_preds[2] += rf.predict_proba(X_val_scaled)[:, 1] / CFG["n_folds"]
        test_preds[2] += rf.predict_proba(X_test_scaled)[:, 1] / CFG["n_folds"]

    print("\n--- Validation Ensemble AUC Scores ---")
    for name, p in zip(MODEL_NAMES, val_preds):
        print(name, "AUC:", roc_auc_score(y_val, p))

    return val_preds, test_preds


# ==============================================================================
# WEIGHT SEARCH
# ==============================================================================

def weighted(p, w):
    return sum(w[i] * p[i] for i in range(len(p)))

def search_weights(val_preds, y_val):
    rng = np.random.default_rng(CFG["random_seed"])
    best = {"w": None, "score": -1}

    for _ in tqdm(range(CFG["n_iterations"]), desc="Optimizing Weights (FN-Optimized)"):
        w = rng.dirichlet(np.ones(len(val_preds)))
        prob = weighted(val_preds, w)

        # Kekemeleri kaçırmamak adına eşiği 0.4 yerine daha hassas (0.35) simüle edelim
        f1 = f1_score(y_val, prob > 0.35, average="macro")  
        rec = recall_score(y_val, prob > 0.35)
        
        # Algoritmayı hem F1'i yüksek tutmaya hem de RECALL'u (kekemeleri yakalamaya) zorluyoruz
        score = f1 

        if score > best["score"]:
            best = {"w": w, "score": score}

    print("Best Ensemble Weights Weight Score (Val)", best["score"])
    return best


# ==============================================================================
# THRESHOLD
# ==============================================================================

def find_threshold(val_preds, y_val, w):
    prob = weighted(val_preds, w)
    best_t, best_score = 0.5, -1

    # Eşik değerini daha yukarıya (0.40 - 0.75 arası) çekiyoruz ki model kolay kolay "Kekeme" diyemesin
    for t in np.linspace(0.40, 0.75, 100):   
        pred = prob > t
        f1 = f1_score(y_val, pred, average="macro")
        
        # Sadece saf Makro F1'e odaklanıyoruz. Bu, iki sınıfı otomatik olarak dengeler.
        score = f1 

        if score > best_score:
            best_score = score
            best_t = t

    print("Best threshold (Val):", best_t)
    return best_t


# ==============================================================================
# FINAL EVAL (ON TEST SET)
# ==============================================================================

def evaluate(test_preds, y_test, w, t):
    prob = weighted(test_preds, w)
    pred = prob > t

    auc_score = roc_auc_score(y_test, prob)

    print("\n🚀 FINAL TEST RESULTS")
    print("========================")
    print("Acc :", accuracy_score(y_test, pred))
    print("F1  :", f1_score(y_test, pred, average="macro"))
    print("Rec :", recall_score(y_test, pred))
    print("AUC :", auc_score)
    print("========================\n")

    print(classification_report(y_test, pred, target_names=CLASS_NAMES))

    plt.figure(figsize=(12, 5))
    
    # 1. Grafiği Çizdirme: Normalize Confusion Matrix (Hata Düzeltildi)
    plt.subplot(1, 2, 1)
    # Doğru Yöntem: Normalizasyon burada, confusion_matrix oluşturulurken yapılır!
    cm = confusion_matrix(y_test, pred, normalize="true") 
    
    # .plot() içerisinden hatalı 'normalize' parametresi kaldırıldı, kararlı hale getirildi.
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", ax=plt.gca(), values_format=".2f")
    plt.title("Ensemble Final Normalized Confusion Matrix (Test Set)")

    # 2. Grafiği Çizdirme: ROC-AUC Eğrisi
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Ensemble ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--") 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Özgüllük)")
    plt.ylabel("True Positive Rate (Duyarlılık / Recall)")
    plt.title("Ensemble ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    return pred, prob

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # 1. Verileri yükle



    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2. Modelleri eğit 
    val_preds, test_preds = train_and_infer(X_train, y_train, X_val, y_val, X_test)

    # 3. Ağırlıkları optimize et
    best = search_weights(val_preds, y_val)

    # 4. En iyi eşik değerini bul
    threshold = find_threshold(val_preds, y_val, best["w"])

    # 5. Final Değerlendirmeyi yap
    evaluate(test_preds, y_test, best["w"], threshold)


if __name__ == "__main__":
    main()