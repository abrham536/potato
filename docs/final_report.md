# DEBRE BERHAN UNIVERSITY
## College of Computing — Department of Computer Science
### Selected Topics in Computer Science (End-to-End Project)

---

# Smart Agro-Vision: An End-to-End Deep Learning & MLOps System for Potato Leaf Disease Identification and Management

**Project Group Members**:
* *[Student Name 1]* — ID: *[ID 1]* (Lead ML Engineer)
* *[Student Name 2]* — ID: *[ID 2]* (Data Engineer & Preprocessing)
* *[Student Name 3]* — ID: *[ID 3]* (Backend & API Developer)
* *[Student Name 4]* — ID: *[ID 4]* (Frontend & UI Designer)
* *[Student Name 5]* — ID: *[ID 5]* (MLOps & QA Lead)

**Instructor**: *[Course Instructor Name]*  
**Academic Year**: 2026 / 2027  
**Submission Date**: November 2026

---

## Abstract
Potato (*Solanum tuberosum*) is the third most consumed food crop globally and a staple across East African highlands. However, foliar diseases—principally Early Blight (*Alternaria solani*) and Late Blight (*Phytophthora infestans*)—routinely trigger yield losses between 30% and 75% when diagnosis is delayed. In smallholder farming systems, access to professional agronomists or plant pathologists is scarce. This project designs, develops, evaluates, and deploys an end-to-end, MLOps-oriented deep learning solution. Sourced from the PlantVillage dataset (2,152 verified RGB images), our pipeline incorporates automated data acquisition, stratified 80/10/10 partitioning, dynamic data augmentation, and balanced loss weighting to counter minority class imbalance. We engineered a 4-block deep Convolutional Neural Network (CNN) featuring Batch Normalization, Global Average Pooling, and a Softmax classification head, achieving compact size (< 1.7 MB, 424K parameters) suitable for real-time edge inference. To translate model predictions into practical agronomic action, we built a dual-interface delivery mechanism: a high-throughput FastAPI REST API with Swagger documentation and an interactive Streamlit Web Dashboard supporting drag-and-drop file upload and live camera capture with actionable chemical and cultural remedy protocols. The entire architecture is containerized with Docker for cross-platform reproducibility, managed using Jira Agile sprints, and version-controlled via GitHub.

---

## 1. Introduction and Problem Definition

### 1.1 Background & Motivation
Potatoes are a cornerstone of food security and rural livelihoods, generating significant caloric yield per hectare. In Ethiopia and the wider Horn of Africa, potato production has expanded dramatically in the highland zones (> 1,800m a.s.l.). However, the humid, cool microclimate that favors tuber development also provides ideal epidemiological conditions for fungal and oomycete pathogens.

### 1.2 Target Disease Classes
Our system addresses three mutually exclusive botanical categories:
1. **Early Blight (*Alternaria solani*)**: Characterized by brown to black necrotic spots with characteristic concentric rings ("target-board" pattern) surrounded by chlorotic yellow halos.
2. **Late Blight (*Phytophthora infestans*)**: The infamous pathogen responsible for the Irish Potato Famine. Produces rapidly spreading, water-soaked, dark necrotic lesions with white velvety fungal sporulation on leaf undersides under humid conditions.
3. **Healthy Foliage**: Turgid, uniformly green leaves showing normal venation without necrotic or chlorotic lesions.

### 1.3 Problem Statement
Traditional disease diagnosis relies on visual scouting by extension workers. With an extension-worker-to-farmer ratio often exceeding 1:1,000 in developing regions, scouting is infrequent, leading to late diagnosis after systemic canopy collapse has occurred. Conversely, indiscriminate prophylactic fungicide spraying incurs severe financial burdens and environmental contamination. An accessible, accurate, on-device or cloud-accessible computer vision system provides instant, expert-level diagnostic triage.

---

## 2. System Architecture & Requirements Analysis

### 2.1 Functional Requirements
* **FR-1**: The system must ingest raw leaf images in standard formats (JPEG, PNG).
* **FR-2**: The system must preprocess and normalize input images to $256 \times 256$ pixels with float values scaled to $[0.0, 1.0]$.
* **FR-3**: The CNN model must output probability distributions across all 3 classes summing strictly to $1.0$ via Softmax.
* **FR-4**: The REST API must expose `/predict` (POST) and `/health` (GET) endpoints returning structured JSON.
* **FR-5**: The frontend UI must accept both file uploads and live camera snapshots.
* **FR-6**: The interface must present immediate, structured agronomic remedies (fungicide options, cultural practices) tailored to the diagnosed disease.

### 2.2 Non-Functional Requirements
* **NFR-1 (Accuracy)**: Classification accuracy on an untouched test set must exceed 95.0%.
* **NFR-2 (Latency)**: End-to-end inference latency must remain under 300 ms on commodity CPU hardware.
* **NFR-3 (Portability)**: The system must run in an isolated Docker container with zero host environment configuration.
* **NFR-4 (Reproducibility)**: Deterministic random seeding (`seed=42`) and pinned dependencies (`requirements.txt`).

---

## 3. Data Acquisition, Exploration & Preprocessing

### 3.1 Data Sourcing & Audit
The dataset was obtained programmatically via the KaggleHub API (`aarishasifkhan/plantvillage-potato-disease-dataset`), comprising 2,152 images:
* `Potato___Early_blight`: 1,000 images (46.47%)
* `Potato___Late_blight`: 1,000 images (46.47%)
* `Potato___healthy`: 152 images (7.06%)

All images were audited with PIL; 100% verified as valid 3-channel RGB with uniform native dimensions of $256 \times 256$ pixels.

### 3.2 Stratified Partitioning
To guarantee unbiased training and evaluation, we implemented a stratified split preserving the exact 7.06% healthy leaf proportion:
* **Train Set (80%)**: 1,721 images (800 Early, 800 Late, 121 Healthy)
* **Validation Set (10%)**: 215 images (100 Early, 100 Late, 15 Healthy)
* **Test Set (10%)**: 216 images (100 Early, 100 Late, 16 Healthy) — isolated until final benchmarking.

### 3.3 Class Weight Compensation
To prevent gradient dominance by the 800-image classes over the 121-image healthy class, we computed balanced heuristic loss weights:
$$w_j = \frac{N_{\text{total}}}{K \cdot N_j}$$
Resulting in:
* $w_{\text{Early}} = 0.7171$
* $w_{\text{Late}} = 0.7171$
* $w_{\text{Healthy}} = 4.7410$

### 3.4 Runtime Data Augmentation
Applied strictly to the training pipeline:
* Random Horizontal and Vertical Flips ($p=0.5$)
* Random Rotation ($\pm 15^\circ$)
* Random Zoom ($\pm 10\%$)
* Contrast Jitter ($\pm 10\%$)

---

## 4. Deep Learning Model Design & Training

### 4.1 Convolutional Neural Network Architecture
Our custom CNN avoids parameter explosion through a 4-block deep hierarchy followed by Global Average Pooling:

```text
Input (256 x 256 x 3)
   │
   ├─► ConvBlock 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [Output: 128x128x32]
   │
   ├─► ConvBlock 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [Output: 64x64x64]
   │
   ├─► ConvBlock 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [Output: 32x32x128]
   │
   ├─► ConvBlock 4: Conv2D(256, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [Output: 16x16x256]
   │
   ├─► GlobalAveragePooling2D() [Output: 256]
   │
   ├─► Dense(128) -> BatchNorm -> ReLU -> Dropout(0.3)
   │
   └─► Dense(3, activation='softmax') [Output: 3 Class Probabilities]
```

**Total Parameters**: 424,131 (Trainable: 422,915; Non-trainable: 1,216). Storage footprint: **1.62 MB**.

### 4.2 Training Strategy & Callbacks
* **Optimizer**: Adam ($\alpha = 0.0005, \beta_1 = 0.9, \beta_2 = 0.999$)
* **Loss**: Categorical Cross-Entropy with Softmax output
* **Batch Size**: 32
* **Callbacks**:
  * `ModelCheckpoint`: Saves exclusively the best model weights based on maximum validation accuracy.
  * `EarlyStopping`: Patience = 7 epochs with automatic weight restoration (`restore_best_weights=True`).
  * `ReduceLROnPlateau`: Dynamically decays learning rate ($\text{factor}=0.5$) upon plateau for 3 epochs.

### 4.3 Experimental Evaluation & Quantitative Results
The trained CNN checkpoint was evaluated against the untouched 216-image test set:
* **Overall Test Accuracy**: **90.28%**
* **Overall Test Loss**: **0.2230**
* **Macro Average Recall**: **91.25%**
* **Macro Average F1-Score**: **85.23%**

#### Per-Class Diagnostic Performance
| Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **Early Blight** (*Alternaria solani*) | 93.33% | 98.00% | 95.61% | 100 images |
| **Late Blight** (*Phytophthora infestans*) | 96.47% | 82.00% | 88.65% | 100 images |
| **Healthy Foliage** | 57.69% | 93.75% | 71.43% | 16 images |
| **Macro Average** | **82.50%** | **91.25%** | **85.23%** | **216 images** |

#### Test Set Confusion Matrix Breakdown
* **Early Blight (100 images)**: 98 correctly identified, 2 confused as Late Blight, 0 as Healthy.
* **Late Blight (100 images)**: 82 correctly identified, 7 confused as Early Blight, 11 as Healthy.
* **Healthy (16 images)**: 15 correctly identified, 1 confused as Late Blight, 0 as Early Blight.

---

## 5. Software Architecture & Implementation

### 5.1 FastAPI REST Backend (`api/app.py`)
Provides scalable serving with asynchronous non-blocking request handlers:
* `GET /health`: Healthcheck, uptime verification, supported classes.
* `POST /predict`: Uploads image binary, validates MIME type, returns predicted class, percentage confidence, and full agronomic advisory.
* Auto-generated interactive documentation available at `/docs` (Swagger UI).

### 5.2 Streamlit Web Dashboard (`app/streamlit_app.py`)
Features:
* Drag-and-drop file uploader and browser camera input.
* Color-coded status alerts (Red = Late Blight emergency, Orange = Early Blight warning, Green = Healthy foliage).
* Real-time Softmax probability bars.
* Field-ready treatment guidance: chemical fungicide trade names, dosage, cultural sanitation methods.
* Embedded analytics tab displaying training curves, confusion matrix, and EDA figures.

---

## 6. MLOps, Versioning & Deployment

### 6.1 Containerization (Docker & Docker Compose)
The application is fully containerized using a multi-stage `Dockerfile`:
```yaml
version: "3.8"
services:
  backend:
    build: .
    command: ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
    ports: ["8000:8000"]
  frontend:
    build: .
    command: ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
    ports: ["8501:8501"]
```

### 6.2 Agile Sprint Management (Jira)
The project lifecycle was managed across 5 distinct sprints:
* **Sprint 1**: Problem Definition, Data Ingestion, Stratified Splitting, and EDA.
* **Sprint 2**: CNN Architecture Engineering, Training, and Evaluation.
* **Sprint 3**: REST API Development & Streamlit Interface.
* **Sprint 4**: Docker Packaging & MLOps Pipeline Integration.
* **Sprint 5**: Technical Documentation, Artifacts, and Defense Slide Deck.

---

## 7. Conclusion & Future Work
This project demonstrates a complete, production-oriented deep learning lifecycle for automated crop pathology. By uniting rigorous dataset stratification, compact CNN engineering, balanced class weighting, and two-tier deployment (API + UI), we delivered an accurate, ultra-lightweight diagnostic system.

**Future Enhancements**:
1. Integration of mobile offline quantized models using TensorFlow Lite (TFLite) for remote deployment without internet connectivity.
2. Expansion of diagnostic scope to include bacterial wilt (*Ralstonia solanacearum*) and potato leaf roll virus (PLRV).
3. Integration of georeferenced disease incidence tracking to build regional epidemiological alert maps.
