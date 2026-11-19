"""
Streamlit Cloud Entry Point
Delegates execution to app/streamlit_app.py to ensure zero-configuration deployment.
"""
import sys
from pathlib import Path

# Add project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and execute the main Streamlit application
from app.streamlit_app import main

if __name__ == "__main__":
    main()
