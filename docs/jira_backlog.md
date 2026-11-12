# Jira Backlog & Sprint Planning Guide
## Course: Selected Topics in Computer Science — Debre Berhan University
### Project: Potato Leaf Disease Identification & Classification System

You can copy and paste these exact tickets into your team's **Jira Scrum Board**.

---

## 🏃 Sprint 1: Problem Definition, Data Acquisition & Preprocessing (Status: DONE)

### [STCS-1] Problem Definition and Requirement Specifications
- **Issue Type**: Story
- **Story Points**: 3
- **Assignee**: Group Member 1
- **User Story**:
  *As an* agricultural stakeholder / farmer,
  *I want* an automated system to diagnose potato leaf diseases,
  *So that* crop damage can be mitigated before wide-scale yield loss occurs.
- **Acceptance Criteria**:
  - Clear problem background documented (Early Blight, Late Blight, Healthy).
  - Identification of functional and non-functional requirements.
  - Definition of success metrics (Accuracy $> 95\%$, inference $< 500\text{ms}$).

### [STCS-2] Automated Dataset Acquisition via Kaggle API
- **Issue Type**: Task
- **Story Points**: 2
- **Assignee**: Group Member 2
- **Description**:
  Implement automated script to download and cache the PlantVillage potato disease dataset using `kagglehub`.
- **Acceptance Criteria**:
  - Script downloads 2,152 images directly to `data/raw/`.
  - Zero manual browser downloads required.

### [STCS-3] Exploratory Data Analysis & Quality Audit
- **Issue Type**: Task
- **Story Points**: 3
- **Assignee**: Group Member 3
- **Description**:
  Conduct image dimension verification, channel validation, corruption check, and class balance analysis.
- **Acceptance Criteria**:
  - Confirmed 2,152 valid RGB images ($256 \times 256$).
  - Generated bar chart of class frequencies (`reports/figures/class_distribution.png`).
  - Generated sample comparison grid (`reports/figures/sample_images.png`).

### [STCS-4] Stratified Dataset Splitting & Organization
- **Issue Type**: Task
- **Story Points**: 2
- **Assignee**: Group Member 1
- **Description**:
  Partition dataset into 80% Train, 10% Validation, and 10% Test partitions using stratified sampling.
- **Acceptance Criteria**:
  - Training set has 1,721 images.
  - Validation set has 215 images.
  - Test set has 216 images.
  - Stratification maintains proportional representation of the minority healthy class.

### [STCS-5] Data Preprocessing, Augmentation & Class Weighting
- **Issue Type**: Task
- **Story Points**: 5
- **Assignee**: Group Member 2
- **Description**:
  Build `tf.data` input pipeline with pixel normalization, runtime data augmentation (flips, rotations, zoom), and computed class weights.
- **Acceptance Criteria**:
  - Pixel values normalized to $[0.0, 1.0]$.
  - Class weights calculated ($w_{\text{Early}} = 0.717$, $w_{\text{Healthy}} = 4.741$, $w_{\text{Late}} = 0.717$).
  - Augmentation visual proof generated (`reports/figures/augmentation_samples.png`).

---

## 🏃 Sprint 2: Model Architecture, Training & Evaluation (Status: TO DO / IN PROGRESS)

### [STCS-6] Custom CNN Architecture Design & Implementation
- **Issue Type**: Story
- **Story Points**: 5
- **Assignee**: Group Member 1
- **User Story**:
  *As an* AI developer,
  *I want* to design and implement a custom deep CNN architecture,
  *So that* it can extract multi-scale spatial features from potato leaves with minimal overfitting.
- **Acceptance Criteria**:
  - Architecture contains 4 Convolutional blocks (Conv2D + BatchNormalization + ReLU + MaxPooling2D).
  - GlobalAveragePooling2D / Flatten with Dropout regularization ($p=0.3$).
  - Dense output layer with 3 units and Softmax activation.
  - Total parameter count under 5 million for edge deployment efficiency.

### [STCS-7] Transfer Learning Benchmark (MobileNetV2 / EfficientNet)
- **Issue Type**: Story
- **Story Points**: 5
- **Assignee**: Group Member 2
- **User Story**:
  *As an* AI researcher,
  *I want* to train a pretrained model via Transfer Learning,
  *So that* we can benchmark our custom CNN against state-of-the-art vision models.
- **Acceptance Criteria**:
  - Pretrained ImageNet weights loaded with base layers frozen.
  - Fine-tuning top classification head on potato dataset.
  - Side-by-side performance comparison against custom CNN.

### [STCS-8] Training Loop, Callbacks & Experiment Tracking
- **Issue Type**: Task
- **Story Points**: 3
- **Assignee**: Group Member 3
- **Description**:
  Train models using Adam optimizer, categorical cross-entropy loss, and balanced class weights with automated callbacks.
- **Acceptance Criteria**:
  - `ModelCheckpoint` saves best model weights based on validation loss.
  - `EarlyStopping` prevents overfitting (patience=5).
  - `ReduceLROnPlateau` dynamically adapts learning rate.

### [STCS-9] Comprehensive Model Evaluation & Diagnostic Curves
- **Issue Type**: Task
- **Story Points**: 3
- **Assignee**: Group Member 1
- **Description**:
  Evaluate best model on untouched Test partition (216 images) and produce academic figures.
- **Acceptance Criteria**:
  - Training vs. Validation Loss and Accuracy curves plotted.
  - Normalized Confusion Matrix generated.
  - Classification Report with Precision, Recall, F1-Score per class.
  - Test Accuracy target: $\ge 96\%$.

---

## 🏃 Sprint 3: REST API & Front-End Dashboard (Status: UPCOMING)

### [STCS-10] FastAPI Backend Service & Prediction Endpoints
- **Issue Type**: Story
- **Story Points**: 5
- **User Story**:
  *As a* client application / developer,
  *I want* a fast REST API endpoint (`/predict`),
  *So that* external systems can query leaf diagnoses with low latency.
- **Acceptance Criteria**:
  - Accepts image file via multipart upload.
  - Returns JSON response with predicted class, confidence percentage, and disease advice.
  - Auto-generated Swagger documentation accessible at `/docs`.

### [STCS-11] Interactive Streamlit Dashboard for Agricultural End-Users
- **Issue Type**: Story
- **Story Points**: 5
- **User Story**:
  *As a* farm scout or agricultural officer,
  *I want* an intuitive web dashboard,
  *So that* I can upload leaf photos and receive immediate diagnosis and treatment recommendations.
- **Acceptance Criteria**:
  - Clean UI with drag-and-drop image upload.
  - Visual confidence gauge across all 3 classes.
  - Actionable agronomic advice (fungicide options, cultural control practices).

---

## 🏃 Sprint 4: MLOps, Containerization & Deployment (Status: UPCOMING)

### [STCS-12] Docker Containerization & Multi-Stage Build
- **Issue Type**: Task
- **Story Points**: 3
- **Acceptance Criteria**:
  - `Dockerfile` packaged with lightweight Python base.
  - Single command startup (`docker run -p 8501:8501 ...`).

### [STCS-13] Deployment & Evidence Documentation
- **Issue Type**: Task
- **Story Points**: 3
- **Acceptance Criteria**:
  - Application deployed locally and/or cloud platform.
  - Screenshots and execution logs captured as deployment evidence.

---

## 🏃 Sprint 5: Technical Documentation & Final Defense (Status: UPCOMING)

### [STCS-14] Final Technical Report & API Documentation
- **Issue Type**: Task
- **Story Points**: 5
- **Acceptance Criteria**:
  - Complete university-standard technical report covering all phases.
  - GitHub repository finalized with comprehensive `README.md`.

### [STCS-15] Final Defense Presentation Slide Deck
- **Issue Type**: Task
- **Story Points**: 3
- **Acceptance Criteria**:
  - Slide deck prepared for faculty review and demo.
