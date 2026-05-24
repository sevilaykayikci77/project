## Label Processing and Dataset Standardization(label_processor.py)

Bu script, SEP-28k ve FluencyBank veri setlerini ortak bir yapıda birleştirerek güvenilir kekemelik etiketleri oluşturmak için geliştirilmiştir.  
Amaç, farklı veri kaynaklarını tek bir standart format altında toplamak ve binary sınıflandırma için temiz bir veri seti üretmektir.

---

## Scriptin Yaptıkları

### 1. Veri Setlerini Yükleme

Aşağıdaki etiket dosyaları okunur:

- `SEP-28k_labels.csv`
- `FluencyBank_labels.csv`

Dosya bulunamazsa hata mesajı gösterilir.

---

### 2. FluencyBank Veri Standardizasyonu

FluencyBank veri setindeki sütun isimleri SEP-28k formatına dönüştürülür:

| Eski İsim | Yeni İsim |
|---|---|
| show | Show |
| ep_id | EpId |
| clip_id | ClipId |
| start | Start |
| stop | Stop |

Ayrıca `EpId` alanı sıfır dolgulu (`001`, `002`) formata çevrilir.

---

### 3. Veri Setlerini Birleştirme

İki veri seti ortak sütun yapısına getirildikten sonra:

```python
pd.concat()

## Audio Files Verification(verify_audio_files.py)

Bu script, etiket dosyasında (`processed_labels.csv`) bulunan kayıtların ilgili `.wav` ses dosyalarıyla eşleşip eşleşmediğini doğrulamak için geliştirilmiştir.  
Amaç, eğitim sürecinden önce veri bütünlüğünü kontrol etmek ve eksik ses dosyalarını tespit etmektir.

---

## Scriptin Yaptıkları

### 1. Dosya ve Klasör Kontrolü
Script aşağıdaki kaynakların mevcut olup olmadığını kontrol eder:

- `processed_labels.csv`
- `processed_clips/` klasörü

Eksik olmaları durumunda hata mesajı verir.

---

### 2. Veri Seti Analizi

CSV dosyası okunur ve:

- Toplam kayıt sayısı
- Ses klasöründeki toplam `.wav` dosya sayısı

hesaplanır.

---

### 3. Ses Dosyası Eşleştirme

Her CSV satırı için otomatik olarak ses dosyası adı oluşturulur:

```python
{Show}_{EpId}_{ClipId}.wav
## Kekemelik Tespiti için HuBERT Fine-Tuning (train_transfer_hubert.py)

Bu kod, önceden eğitilmiş HuBERT (`facebook/hubert-base-ls960`) konuşma temsil modeli kullanılarak transfer öğrenme tabanlı kekemelik tespiti gerçekleştirir.

### İşlem Akışı (Pipeline Overview)

Çalışma akışı aşağıdaki aşamalardan oluşmaktadır:

1. **Veri Setinin Yüklenmesi**
   - Eğitim (Train), doğrulama (Validation) ve test kümeleri CSV dosyalarından ayrı ayrı yüklenir.
   - Ses dosyalarının yolları (`Show`, `EpId`, `ClipId`) meta verileri kullanılarak otomatik olarak oluşturulur.

2. **Ses Verisi Ön İşleme**
   - Ses kayıtları `soundfile` kütüphanesi ile okunur.
   - Stereo kayıtlar mono formata dönüştürülür.
   - Bozuk veya okunamayan ses dosyaları otomatik olarak atlanır.
   - Tüm ses örnekleri aşağıdaki standart yapıya dönüştürülür:
     - 16 kHz örnekleme frekansı
     - 3 saniye sabit uzunluk
     - padding/truncation işlemleri

3. **HuBERT Özellik Kodlama**
   - Ham ses dalgaları (`raw waveform`) `Wav2Vec2FeatureExtractor` kullanılarak HuBERT uyumlu giriş temsiline dönüştürülür.

4. **Transfer Öğrenme (Fine-Tuning)**
   - Önceden eğitilmiş `HuBERT Base LS960` modeli ikili sınıflandırma için fine-tune edilir:
     - Sınıf 0 → Akıcı konuşma
     - Sınıf 1 → Kekemelik içeren konuşma
   - Eğitim kararlılığını artırmak için CNN tabanlı feature encoder katmanı dondurulmuştur.
   - Transformer katmanları eğitilebilir durumda bırakılmıştır.

5. **Model Eğitimi**
   - Eğitim işlemi Hugging Face `Trainer` API kullanılarak gerçekleştirilir.
   - CUDA destekli GPU mevcutsa mixed precision (`fp16`) eğitimi otomatik etkinleştirilir.
   - En iyi model doğrulama F1-score metriğine göre seçilir.

6. **Değerlendirme**
   - Nihai performans tamamen bağımsız test kümesi üzerinde değerlendirilir:
     - Accuracy (Doğruluk)
     - F1-score

7. **Modelin Kaydedilmesi**
   - Fine-tune edilmiş HuBERT modeli ve feature extractor daha sonraki:
     - embedding extraction,
     - feature selection,
     - ensemble learning
     aşamalarında kullanılmak üzere kaydedilir.

---

### Kullanılan Teknolojiler

- PyTorch
- Hugging Face Transformers
- HuBERT
- Hugging Face Datasets
- Evaluate
- NumPy
- Pandas

---

### Çıktılar

Kod aşağıdaki çıktıları üretir:

- Fine-tune edilmiş HuBERT sınıflandırma modeli
- Kaydedilmiş tokenizer / feature extractor
- Test Accuracy ve F1-score raporları

Kaydedilen model dizini:

```bash
./best_hubert_stuttering_model
# HuBERT GAP Embedding Extraction(Gap_feature.py)

Bu script, fine-tune edilmiş HuBERT modeli kullanarak konuşma verilerinden 
768 boyutlu GAP (Global Average Pooling) embedding öznitelikleri çıkarır.

Kod; train, validation ve test setlerini birbirinden tamamen bağımsız şekilde işler 
ve tüm öznitelikleri tek bir `.npz` dosyasında saklar. 
Bu yapı özellikle veri sızıntısını (data leakage) önlemek amacıyla tasarlanmıştır.

---

## İş Akışı

### 1. Veri Bölümlerinin Yüklenmesi
Aşağıdaki veri bölümleri ayrı CSV dosyalarından okunur:

- Train set
- Validation set
- Test set

Her ses dosyasının yolu;
`Show`, `EpId` ve `ClipId` bilgileri kullanılarak otomatik oluşturulur.

---

### 2. Ses Verisi Ön İşleme

Ses dosyaları `soundfile` kullanılarak okunur.

Ön işleme adımları:

- Stereo kayıtlar mono formata dönüştürülür
- Çok kısa veya bozuk ses dosyaları filtrelenir
- Sesler:
  - 16 kHz örnekleme oranına
  - 3 saniye uzunluğa
  - sabit giriş boyutuna dönüştürülür

---

### 3. HuBERT Özellik Çıkarımı

Fine-tune edilmiş HuBERT modeli yüklenir:

- `Wav2Vec2FeatureExtractor`
- `HubertModel`

Her ses örneği modelden geçirilerek:

- Son gizli katman (`last_hidden_state`) elde edilir
- Global Average Pooling (Mean Pooling) uygulanır
- Her ses için 768 boyutlu embedding üretilir

---

### 4. Veri Sızıntısını Önleme

Train, validation ve test setleri:

- ayrı ayrı işlenir
- birbirine karıştırılmaz
- ayrı embedding dizileri halinde tutulur

Bu yaklaşım akademik olarak doğru ve güvenilir değerlendirme sağlar.

---

### 5. NPZ Dosyasına Kaydetme

Tüm embedding'ler tek bir sıkıştırılmış `.npz` dosyasına kaydedilir:

```bash
features/hubert_gap_features.npz
# HuBERT Ablation Study — Systematic Machine Learning Evaluation(ablation.py)

Bu script, fine-tune edilmiş HuBERT modelinden çıkarılan 
GAP embedding öznitelikleri üzerinde sistematik bir ablasyon çalışması gerçekleştirir.

Amaç; farklı:

- özellik seçme yöntemleri (feature selection)
- makine öğrenmesi sınıflandırıcıları
- karar eşikleri (threshold)

arasındaki performans farklarını karşılaştırarak 
en başarılı kombinasyonu belirlemektir.

---

# Kullanılan Özellikler

Girdi olarak aşağıdaki dosya kullanılır:
Feature Selection;
HAM	
KBEST	ANOVA tabanlı SelectKBest
LASSO	L1 regularization tabanlı seçim
TREE Random Forest tabanlı özellik önemi
Modeller;
SVM_Lin	Linear SVM
SVM_RBF	RBF Kernel SVM
SVM_Quad	Quadratic Polynomial SVM
SVM_Cub	Cubic Polynomial SVM
RF	Random Forest
XGBoost	Extreme Gradient Boosting
CatBoost	CatBoost Gradient Boosting
*Seçilen optimal threshold kullanılarak test setinde:
Accuracy
ROC-AUC
Sensitivity
Specificity 
hesaplanır.

*HYBRİD.PY SİLİLNECEK.
