import sys
from pathlib import Path

# Ensure root directory is in sys.path so Flask and model modules resolve
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app

# Expose app for Vercel Python runtime
app = app
