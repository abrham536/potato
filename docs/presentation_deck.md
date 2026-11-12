# Final Project Defense Presentation Slide Deck
## Course: Selected Topics in Computer Science
### Debre Berhan University — College of Computing

**Project Title**: Smart Agro-Vision: End-to-End Potato Leaf Disease Identification and Management System  
**Presentation Time**: 15–20 minutes

---

### Slide 1: Title Slide
* **Project Name**: Smart Agro-Vision
* **Subtitle**: An End-to-End Deep Learning & MLOps System for Potato Leaf Disease Diagnosis
* **Department**: Department of Computer Science, Debre Berhan University
* **Team Members**: *[Names and IDs of all group members]*
* **Supervisor**: *[Instructor / Advisor Name]*

---

### Slide 2: Problem Definition & Motivation
* **Global & Local Significance**: Potatoes are a staple food crop in Ethiopia (> 160,000 hectares under cultivation).
* **Foliar Disease Threat**: Early Blight (*A. solani*) and Late Blight (*P. infestans*) cause up to 75% yield loss.
* **The Challenge**: Shortage of agricultural extension specialists; diagnosis is usually made too late.
* **Our Solution**: Real-time AI diagnostic system accessible via smartphone/web with instant treatment advice.

---

### Slide 3: End-to-End System Architecture (Flow Diagram)
* **Visual**:
  `Farmer Leaf Photo -> Streamlit Web UI -> FastAPI REST API -> 4-Block CNN -> Softmax Diagnosis + Agronomic Protocol`
* **Full MLOps Lifecycle**:
  Data Ingestion -> Cleaning & Stratified Splitting -> Preprocessing & Augmentation -> Custom CNN -> API Serving -> UI Dashboard -> Docker Containerization.

---

### Slide 4: Dataset Exploration & Preprocessing
* **Source**: PlantVillage (2,152 audited RGB images).
* **Class Balance**: Early Blight (1,000), Late Blight (1,000), Healthy (152).
* **Stratified 80/10/10 Partitioning**:
  * Train: 1,721 images | Val: 215 images | Test: 216 images.
* **Class Weighting Strategy**: Healthy class weighted at $4.74\times$ to prevent minority underrepresentation.
* **Data Augmentation**: Flips, $\pm 15^\circ$ rotation, zoom, contrast jittering.

---

### Slide 5: Deep Learning Model Architecture
* **Design Philosophy**: High accuracy with lightweight edge-deployable footprint.
* **4 Convolutional Blocks**:
  * `Conv2D(32 -> 64 -> 128 -> 256)` with `BatchNormalization` and `ReLU`.
  * `MaxPooling2D(2, 2)` after each block.
* **Regularization & Output**:
  * `GlobalAveragePooling2D` (reduces parameters to just 424K!).
  * `Dropout(0.3)` on Dense(128).
  * **Softmax Output Head**: Strictly outputs probability distribution summing to 1.0 across all 3 classes.

---

### Slide 6: Model Training & Callbacks
* **Optimizer**: Adam ($\text{lr} = 0.001$).
* **Loss Function**: Categorical Cross-Entropy + Balanced Class Weights.
* **Smart Callbacks**:
  * `ModelCheckpoint`: Auto-saves best model to `models/best_potato_cnn.keras`.
  * `EarlyStopping`: Halts when validation loss plateaus.
  * `ReduceLROnPlateau`: Adapts learning rate dynamically.

---

### Slide 7: Experimental Results & Model Evaluation
* **Metrics on Untouched Test Set (216 images)**:
  * Overall Test Accuracy: **$> 96\%$**
  * Macro F1-Score: **$> 95\%$**
* **Diagnostic Figures Shown**:
  * Training vs. Validation Loss and Accuracy convergence curves.
  * Normalized Confusion Matrix displaying per-class discrimination.

---

### Slide 8: Backend API Architecture (FastAPI)
* **Endpoints**:
  * `GET /health`: Healthcheck, model uptime, supported classes.
  * `POST /predict`: Multipart image upload returning prediction confidence and treatment advice.
* **Interactive Documentation**: Auto-generated OpenAPI / Swagger UI at `/docs`.

---

### Slide 9: User Interface & Dashboard (Streamlit)
* **Features**:
  * Drag-and-drop photo upload + **Live Device Camera capture**.
  * Visual alert badge (Red = Emergency Late Blight, Amber = Early Blight, Green = Healthy).
  * Softmax probability bars.
  * **Agronomic Remedy Guide**: Actionable chemical fungicide trade names, dosage, and cultural practices.

---

### Slide 10: MLOps, Containerization & Task Management
* **Docker Multi-Stage Build**: Runs both frontend and backend via `docker-compose up`.
* **Reproducibility**: `requirements.txt` with frozen dependencies and deterministic seeds.
* **Jira Agile Management**: 5 Sprints with 15 User Stories, story points, and acceptance criteria.
* **GitHub Repository**: Semantic commit history demonstrating individual and team contributions.

---

### Slide 11: Live Demonstration
* *[Switch to live browser demo: Uploading sample healthy leaf, early blight, and late blight image]*
* *[Demonstrate live camera capture feature]*
* *[Demonstrate Swagger API docs at http://localhost:8000/docs]*

---

### Slide 12: Conclusion & Future Scope
* **Key Achievements**:
  * Production-ready, end-to-end AI system built in under 1.7 MB model footprint.
  * Translates raw predictions into tangible agronomic intervention protocols.
* **Future Work**:
  * TFLite mobile offline deployment for remote rural farms without connectivity.
  * Expanding to bacterial wilt and virus detection.
* **Q&A**: Thank you! Open for questions from the examination committee.
