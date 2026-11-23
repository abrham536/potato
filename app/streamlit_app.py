import os
import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    CLASS_NAMES,
    DISPLAY_LABELS,
    FIGURES_DIR,
    REPORTS_DIR,
    MIN_PREDICTION_CONFIDENCE
)
from src.remedies import DISEASE_INFO
from src.potato_verifier import verify_potato_leaf, check_leaf_presence, VerificationResult

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Configure Page
st.set_page_config(
    page_title="Smart Agro-Vision | Potato Disease Classifier",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #27ae60;
        margin-bottom: 15px;
    }
    .alert-card-late {
        background-color: #fdedec;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid #e74c3c;
    }
    .alert-card-early {
        background-color: #fef9e7;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid #f39c12;
    }
    .alert-card-healthy {
        background-color: #eafaf1;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid #2ecc71;
    }
    .alert-card-invalid {
        background-color: #fbeee6;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid #c0392b;
    }
    .alert-card-uncertain {
        background-color: #fff9e6;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid #f39c12;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    if not BEST_MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(str(BEST_MODEL_PATH))

def preprocess_image(image: Image.Image):
    """
    Prepares input tensor for the CNN model.
    Note: The Keras model contains an internal layers.Rescaling(1.0/255.0),
    so raw [0.0, 255.0] float tensor is passed to avoid double rescaling.
    """
    image = image.convert("RGB").resize(IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/potato.png", width=70)
        st.markdown("### **Smart Agro-Vision**")
        st.markdown("**Debre Berhan University**")
        st.markdown("*College of Computing*")
        st.caption("Department of Computer Science — Selected Topics in Computer Science")
        st.divider()

        st.markdown("#### ⚙️ **System Specifications**")
        st.markdown("- **Model**: 4-Block Deep CNN")
        st.markdown("- **Classifier**: Softmax Activation (3-Class)")
        st.markdown("- **Input Tensor**: 256×256×3 RGB")
        st.markdown("- **Inference**: On-Device CPU/GPU")
        st.divider()

        st.markdown("#### 📌 **Navigation**")
        nav_choice = st.radio(
            "Select View:",
            ["📸 Leaf Disease Diagnosis", "📊 Model Performance & EDA", "📖 Agronomic Knowledge Base", "👥 Team & Project Info"]
        )

    # 1. Leaf Disease Diagnosis
    if nav_choice == "📸 Leaf Disease Diagnosis":
        st.markdown("<p class='main-title'>🥔 Potato Leaf Disease Diagnosis & Advisory</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Upload or capture a photo of a potato leaf for instant diagnostic screening and treatment recommendations.</p>", unsafe_allow_html=True)

        model = load_trained_model()
        if model is None:
            st.error("⚠️ Trained model file not found. Please train the model first.")
            return

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### 📤 **Provide Leaf Image**")
            input_mode = st.radio("Choose Input Method:", ["File Upload", "Camera Snapshot"], horizontal=True)
            
            uploaded_file = None
            if input_mode == "File Upload":
                uploaded_file = st.file_uploader("Upload leaf photograph (JPG/PNG)", type=["jpg", "jpeg", "png"])
            else:
                uploaded_file = st.camera_input("Capture leaf using device camera")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Leaf Photograph", use_container_width=True)
                st.caption("ℹ️ *Input validation: Image is checked for foliage morphology before diagnostic inference.*")

        with col2:
            st.markdown("### 🔬 **Diagnostic Results**")
            if uploaded_file is not None:
                with st.spinner("Analyzing botanical morphology and potato leaf manifold..."):
                    # 1. Multi-tier verification: Ensure input is specifically a potato leaf
                    ver_res = verify_potato_leaf(image)

                    if not ver_res.is_valid:
                        if not ver_res.is_leaf:
                            card_title = "🚫 NON-LEAF IMAGE DETECTED"
                            card_desc = "The uploaded photograph does not contain biological plant foliage or leaves."
                            advisory_note = "Non-leaf subjects (human faces, skin, clothing, animals, vehicles, furniture, or random objects) cannot be diagnosed."
                        else:
                            card_title = "🍃 NON-POTATO LEAF DETECTED"
                            card_desc = "Plant leaf detected, but morphology and textural patterns do not match potato foliage (*Solanum tuberosum*)."
                            advisory_note = "This diagnostic system is specialized **exclusively for potato crop foliage**. Leaves from other plants (such as tomato, corn, apple, pepper, or ornamental plants) are rejected to prevent dangerous misdiagnosis."

                        st.markdown(f"""
                        <div class='alert-card-invalid'>
                            <h3 style='margin-top:0; color:#c0392b;'>{card_title}</h3>
                            <p style='font-size:1.15rem; margin-bottom:8px;'>{card_desc}</p>
                            <p style='font-size:0.95rem; color:#7f8c8d; margin-bottom:0;'><b>Diagnostic Detail:</b> {ver_res.reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.warning(f"⚠️ **Diagnostic Halted:** {advisory_note} Please upload a clear photo of a real potato leaf.")

                        with st.expander("🔍 **Verification Telemetry & Diagnostic Metrics**", expanded=False):
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Foliage Coverage", f"{ver_res.foliage_ratio * 100:.1f}%", help="Required >= 18%")
                            m2.metric("Excess Green (ExG)", f"{ver_res.exg_mean:.1f}", help="Required >= 18")
                            m3.metric("Organic Hue Var", f"{ver_res.hue_std:.1f}°", help="Required >= 3.5°")
                            m4.metric("Manifold MSE", f"{ver_res.reconstruction_mse:.4f}", help="Required <= 0.032")
                    else:
                        tensor = preprocess_image(image)
                        probabilities = model(tensor, training=False).numpy()[0]

                        pred_idx = int(np.argmax(probabilities))
                        pred_class = CLASS_NAMES[pred_idx]
                        display_name = DISPLAY_LABELS[pred_idx]
                        confidence = probabilities[pred_idx] * 100

                        st.caption("✅ **Verified Input:** Genuine potato leaf foliage (*Solanum tuberosum*) confirmed.")

                        # Check for low confidence / ambiguous prediction
                        if confidence < MIN_PREDICTION_CONFIDENCE:
                            st.markdown(f"""
                            <div class='alert-card-uncertain'>
                                <h3 style='margin-top:0; color:#d35400;'>⚠️ LOW CONFIDENCE: UNCERTAIN DIAGNOSIS</h3>
                                <p style='font-size:1.15rem; margin-bottom:5px;'>Tentative Prediction: <b>{display_name}</b></p>
                                <p style='font-size:1.0rem; margin-bottom:0;'>Confidence: <b>{confidence:.2f}%</b> (Below reliable threshold of {MIN_PREDICTION_CONFIDENCE:.0f}%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.info("💡 **Recommendation:** The leaf features are ambiguous or blurry. Please take a closer photograph with natural daylight focused on the leaf surface.")
                        else:
                            # Styling card based on disease
                            if pred_class == "Potato___Late_blight":
                                card_class = "alert-card-late"
                                badge = "🚨 CRITICAL DANGER: LATE BLIGHT DETECTED"
                            elif pred_class == "Potato___Early_blight":
                                card_class = "alert-card-early"
                                badge = "⚠️ WARNING: EARLY BLIGHT DETECTED"
                            else:
                                card_class = "alert-card-healthy"
                                badge = "✅ EXCELLENT: HEALTHY LEAF"

                            st.markdown(f"""
                            <div class='{card_class}'>
                                <h3 style='margin-top:0;'>{badge}</h3>
                                <p style='font-size:1.2rem; margin-bottom:5px;'>Predicted Condition: <b>{display_name}</b></p>
                                <p style='font-size:1.1rem; margin-bottom:0;'>Confidence Level: <b>{confidence:.2f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("#### **Softmax Probability Distribution**")
                        for name, prob in zip(DISPLAY_LABELS, probabilities):
                            pct = float(prob * 100)
                            st.write(f"**{name}**: {pct:.2f}%")
                            st.progress(min(pct / 100.0, 1.0))

                        # Agronomic Advisory Section
                        remedy = DISEASE_INFO.get(pred_class, {})
                        st.divider()
                        st.markdown("### 🌾 **Agronomic Treatment Protocol**")

                        with st.expander("🔍 **Diagnostic Symptoms**", expanded=True):
                            for s in remedy.get("symptoms", []):
                                st.markdown(f"- {s}")

                        with st.expander("🧪 **Chemical Fungicide Control**", expanded=(pred_class != "Potato___healthy")):
                            for c in remedy.get("chemical_control", []):
                                st.markdown(f"- {c}")

                        with st.expander("🌱 **Cultural & Preventive Practices**", expanded=False):
                            for cp in remedy.get("cultural_practices", []):
                                st.markdown(f"- {cp}")

            else:
                st.info("👈 Please upload or capture an image on the left panel to begin diagnostic analysis.")

    # 2. Model Performance & EDA
    elif nav_choice == "📊 Model Performance & EDA":
        st.markdown("<p class='main-title'>📊 Model Performance & Dataset Exploration</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Audited dataset properties, training convergence curves, and evaluation metrics on unseen test data.</p>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Learning Curves & Confusion Matrix", "Dataset Exploration", "Augmentation Pipeline"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### **Training & Validation Curves**")
                curves_path = FIGURES_DIR / "training_curves.png"
                if curves_path.exists():
                    st.image(str(curves_path), use_container_width=True)
                else:
                    st.info("Training curves will appear once model training finishes.")
            with c2:
                st.markdown("#### **Confusion Matrix (Test Set)**")
                cm_path = FIGURES_DIR / "confusion_matrix.png"
                if cm_path.exists():
                    st.image(str(cm_path), use_container_width=True)
                else:
                    st.info("Confusion matrix will appear after running evaluation.")

        with tab2:
            st.markdown("#### **Class Frequencies & Raw Leaf Samples**")
            c1, c2 = st.columns([1, 1])
            with c1:
                dist_path = FIGURES_DIR / "class_distribution.png"
                if dist_path.exists():
                    st.image(str(dist_path), caption="Dataset Class Distribution", use_container_width=True)
            with c2:
                sample_path = FIGURES_DIR / "sample_images.png"
                if sample_path.exists():
                    st.image(str(sample_path), caption="Raw Sample Leaves across Conditions", use_container_width=True)

        with tab3:
            st.markdown("#### **Data Augmentation Transformations**")
            aug_path = FIGURES_DIR / "augmentation_samples.png"
            if aug_path.exists():
                st.image(str(aug_path), caption="Random Flips, Rotations, Zoom, and Contrast Jittering", use_container_width=True)

    # 3. Agronomic Knowledge Base
    elif nav_choice == "📖 Agronomic Knowledge Base":
        st.markdown("<p class='main-title'>📖 Potato Pathology Knowledge Base</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Detailed etiology, life-cycle, and disease management strategies for common Solanum tuberosum pathogens.</p>", unsafe_allow_html=True)

        for cls_name, info in DISEASE_INFO.items():
            with st.expander(f"📌 **{info['common_name']}** (*{info['causal_organism']}*)", expanded=(cls_name == "Potato___Late_blight")):
                st.markdown(f"**Severity Level**: `{info['severity']}`")
                st.markdown("#### Symptoms:")
                for sym in info["symptoms"]:
                    st.markdown(f"- {sym}")
                st.markdown("#### Chemical Control:")
                for chem in info["chemical_control"]:
                    st.markdown(f"- {chem}")
                st.markdown("#### Cultural & Preventive Controls:")
                for cult in info["cultural_practices"]:
                    st.markdown(f"- {cult}")

    # 4. Team & Project Info
    elif nav_choice == "👥 Team & Project Info":
        st.markdown("<p class='main-title'>👥 Project Team & Academic Scope</p>", unsafe_allow_html=True)
        st.markdown("""
        ### **Debre Berhan University**
        **College of Computing — Department of Computer Science**  
        *Course: Selected Topics in Computer Science (End-to-End Project)*

        ---

        #### **Project Title**:
        **Smart Agro-Vision: End-to-End Potato Leaf Disease Identification and Management System**

        #### **Key MLOps Workflow Components Demonstrated**:
        1. **Problem Definition & Requirements Analysis**: Food security impact & disease scouting challenges.
        2. **Automated Data Acquisition**: KaggleHub API pipeline (`download_data.py`).
        3. **Data Preprocessing & Augmentation**: Stratified 80/10/10 split, normalized tensor pipeline, balanced class weights.
        4. **Deep Learning Model Selection**: Custom 4-block CNN architecture with Global Average Pooling and Softmax.
        5. **API & Integration**: High-performance FastAPI REST API (`/predict`, `/health`).
        6. **Interactive Dashboard**: Modern Streamlit Web UI with live camera capture and agronomic advisory.
        7. **Containerization & Deployment**: Dockerized container deployment.
        8. **Version Control & Task Tracking**: Git commit history and Jira Agile Sprint backlog.
        """)

if __name__ == "__main__":
    main()
