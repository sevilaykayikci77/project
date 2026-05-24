import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    Wav2Vec2FeatureExtractor, 
    HubertForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. AYARLAR VE YOL TANIMLAMALARI
# ==========================================
AUDIO_DIR = r"C:\Users\BIL MUH\Desktop\Damla\project_stutter\archive\processed_clips"
CSV_BASE_DIR = r"C:\Users\BIL MUH\Desktop\Damla\project_stutter\archive"

TRAIN_CSV = os.path.join(CSV_BASE_DIR, "train_split.csv")
VAL_CSV   = os.path.join(CSV_BASE_DIR, "val_split.csv")
TEST_CSV  = os.path.join(CSV_BASE_DIR, "test_split.csv")

MODEL_NAME = "facebook/hubert-base-ls960" 
BATCH_SIZE = 8
EPOCHS = 3

# ==========================================
# 2. VERİ ÖN İŞLEME VE METRİKLER
# ==========================================
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def load_local_dataset(csv_path):
    df = pd.read_csv(csv_path)
    file_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        show = str(row['Show']).strip()
        ep_id = str(row['EpId']).strip()
        clip_id = str(row['ClipId']).strip()
        
        if "fluencybank" in show.lower():
            save_ep_id = ep_id.zfill(3)
        else:
            save_ep_id = ep_id
            
        filename = f"{show}_{save_ep_id}_{clip_id}.wav"
        full_path = os.path.join(AUDIO_DIR, filename)
        
        if os.path.exists(full_path):
            file_paths.append(full_path)
            labels.append(int(row['is_stutter']))

    print(f"📈 {os.path.basename(csv_path)} içinden {len(file_paths)} geçerli ses dosyası yüklendi.")
    return Dataset.from_dict({"file": file_paths, "label": labels})


def preprocess_audio_data(examples):
    """
    Batched=True moduna tam uyumlu, bozuk dosyaları filtreleyen 
    ve HuBERT özniteliklerini toplu üreten akıllı fonksiyon.
    """
    audio_inputs = []
    valid_labels = []
    
    # batched=True olduğu için examples["file"] ve examples["label"] birer listedir.
    for path, label in zip(examples["file"], examples["label"]):
        try:
            speech, sr = sf.read(path)
            if len(speech.shape) > 1:
                speech = speech[:, 0]  # Stereo ise Mono'ya çevir
            
            # Ses verisi başarılı okunduysa listeye ekle
            if len(speech) > 0:
                audio_inputs.append(speech)
                valid_labels.append(label)
        except Exception as e:
            # Okunamayan, bozuk veya eksik dosyaları doğrudan pas geçiyoruz
            continue
            
    # Eğer batch'teki tüm dosyalar bir şekilde bozuk çıktıysa (düşük ihtimal) 
    # HuggingFace çökmesin diye güvenli bir sessiz dolgu yapıyoruz.
    if not audio_inputs:
        audio_inputs = [np.zeros(16000 * 3)]
        valid_labels = [0]

    # Ses sinyallerini HuBERT formatına (16kHz, 3 saniye sabitleme) dönüştürüyoruz
    inputs = feature_extractor(
        audio_inputs, 
        sampling_rate=16000, 
        max_length=int(16000 * 3.0), 
        truncation=True, 
        padding="max_length",
        return_attention_mask=True
    )
    
    # Temizlenmiş ve senkronize edilmiş etiketleri geri veriyoruz
    inputs["label"] = valid_labels
    return inputs


metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = metric_accuracy.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="binary")
    return {**acc, **f1}

# ==========================================
# 3. TRANSFER LEARNING VE EĞİTİM
# ==========================================
def main():
    print("🔄 Veri setleri yükleniyor...")
    raw_datasets = DatasetDict({
        "train": load_local_dataset(TRAIN_CSV),
        "validation": load_local_dataset(VAL_CSV),
        "test": load_local_dataset(TEST_CSV)
    })

    print("📊 Ses dalgaları HuBERT formatına dönüştürülüyor...")
    # Hız ve kararlılık için batched=True yapıldı. batch_size=1000 idealdir, 
    # RAM'i şişirmeden veriyi biner biner işleyip HuggingFace formatına hızlıca çevirir.
    encoded_datasets = raw_datasets.map(
        preprocess_audio_data, 
        batched=True,
        batch_size=1000,
        remove_columns=["file"]
    )

    print("🏗️ Model yükleniyor...")
    model = HubertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"🚀 Eğitim donanımı: {device.upper()}")

    # CNN tabanlı feature encoder katmanını donduruyoruz, Transformer'lar serbest.
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir="./hubert-stuttering-model",
        eval_strategy="epoch",       
        save_strategy="epoch",
        learning_rate=2e-5,  
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,          
        weight_decay=0.01,
        logging_steps=50,            
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), # CUDA (GPU) varsa performansı katlar
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    print("\n🏋️ HuBERT Fine-Tuning Başlıyor...")
    trainer.train()

    print("\n🧪 Test Seti Üzerinde Son Değerlendirme...")
    test_results = trainer.evaluate(encoded_datasets["test"])
    print(f"\n🎯 SAF DEELEARNING MODELİ TEST SONUÇLARI:")
    print(f"Test Doğruluğu (Accuracy): {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Skoru           : {test_results['eval_f1']:.4f}")

    trainer.save_model("./best_hubert_stuttering_model")
    feature_extractor.save_pretrained("./best_hubert_stuttering_model")
    print("💾 Model başarıyla kaydedildi!")

if __name__ == "__main__":
    main()