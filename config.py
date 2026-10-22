import os
from pathlib import Path

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent

# Directory configurations
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

# Maximum allowed file size for uploads (10 Megabytes)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB in bytes

# Model configuration
# Ultralytics will auto-download 'yolo11n.pt' on first run if not present locally
MODEL_PATH = os.path.join(BASE_DIR, 'yolo11n.pt')
DEFAULT_CONF_THRESHOLD = 0.25
MIN_CONF_THRESHOLD = 0.05
MAX_CONF_THRESHOLD = 1.0

# Secret key for Flask session / flash messages
SECRET_KEY = os.environ.get('SECRET_KEY', 'hemanth-internship-secure-key-2025')

# Server settings
HOST = '127.0.0.1'
PORT = 5000
DEBUG = False

# Ensure required runtime folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
