import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Redirect runtime directories to project workspace to guarantee sandbox permission compliance
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".keras").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "PlantVillage"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Specs
IMAGE_SIZE = (256, 256)
CHANNELS = 3
INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 3
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
DISPLAY_LABELS = ["Early Blight", "Late Blight", "Healthy"]

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 15
INITIAL_LEARNING_RATE = 1e-3  # 0.001
MIN_LEARNING_RATE = 1e-6
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.2
DROPOUT_RATE = 0.3

# Best Model Weights Path
BEST_MODEL_PATH = MODELS_DIR / "best_potato_cnn.keras"
TRAINING_LOG_PATH = REPORTS_DIR / "training_history.csv"
