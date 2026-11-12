# Potato Leaf Disease Identification & Classification (CNN)

An end-to-end Deep Learning system to detect and classify potato leaf diseases using Convolutional Neural Networks (CNN).

## 📊 Dataset Overview
The dataset contains **2,152 leaf images** categorized into 3 conditions:
- **Early Blight** (`Alternaria solani`): 1,000 images
- **Late Blight** (`Phytophthora infestans`): 1,000 images
- **Healthy**: 152 images

- **Image Specifications**: $256 \times 256$ pixels, 3 RGB channels.
- **Source**: PlantVillage (`aarishasifkhan/plantvillage-potato-disease-dataset`) via KaggleHub.

---

## 🗂️ Project Directory Structure

```text
deep project/
├── data/
│   ├── raw/
│   │   └── PlantVillage/          # Original downloaded images
│   └── processed/
│       ├── train/                 # 80% (1,721 images)
│       ├── val/                   # 10% (215 images)
│       └── test/                  # 10% (216 images)
├── reports/
│   └── figures/
│       ├── class_distribution.png # Bar chart of class frequencies
│       ├── sample_images.png      # 3x3 grid of raw leaf samples
│       └── augmentation_samples.png# Transformations preview
├── src/
│   ├── __init__.py
│   ├── data_split.py              # Stratified train/val/test splitter
│   ├── preprocess.py              # TF data pipeline & class weights
│   ├── dataset_pytorch.py         # PyTorch DataLoader pipeline
│   └── visualize_augmentation.py  # Visual augmentation generator
├── download_data.py               # KaggleHub download script
├── explore_data.py                # Dataset audit and shape inspection
├── visualize_data.py              # EDA plotting script
└── README.md
```

---

## ⚙️ Completed Work (Phase 1 & Phase 2)

1. **Automated Ingestion**:
   - Downloaded and extracted dataset from Kaggle via `download_data.py`.
2. **Exploratory Data Analysis (EDA)**:
   - Full audit verifying resolution ($256 \times 256$), RGB color channels, and zero corrupt files.
   - Generated class distribution and sample leaf comparisons in `reports/figures/`.
3. **Stratified Splitting**:
   - Created reproducible 80% / 10% / 10% partition across all 3 classes:
     - **Train**: 800 Early Blight, 121 Healthy, 800 Late Blight (**1,721 total**)
     - **Validation**: 100 Early Blight, 15 Healthy, 100 Late Blight (**215 total**)
     - **Test**: 100 Early Blight, 16 Healthy, 100 Late Blight (**216 total**)
4. **Data Preprocessing & Augmentation**:
   - Rescaling pixel intensity to $[0, 1]$.
   - Data augmentation for training: Random horizontal/vertical flips, $\pm 15^\circ$ rotations, zoom, contrast jitter.
   - Calculated balanced class weights ($w_{\text{Early Blight}} = 0.717$, $w_{\text{Healthy}} = 4.741$, $w_{\text{Late Blight}} = 0.717$) to mitigate class imbalance.
   - Built optimized input pipelines with batching (32) and prefetching (`AUTOTUNE`).

---

## 🚀 Next Milestone (Phase 3: CNN Model Architecture)
- Custom CNN design with Conv2D + BatchNormalization + MaxPooling + Dropout + Dense layers.
- Cross-entropy loss function with Adam optimizer.
- Model training with EarlyStopping and ModelCheckpoint.
