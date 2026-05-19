"""
Configuration Settings for CVE NLP Pipeline
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Database
DATABASE_URL = 'sqlite:///./cve_database.db'

# Model Settings
BERT_MODEL = 'bert-base-uncased'

# Device (GPU if available, else CPU)
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training Parameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_SEQ_LENGTH = 512

# Data Collection
NVD_API_BASE_URL = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
NVD_API_KEY = os.getenv('NVD_API_KEY', '')  # Set via environment variable
REQUEST_TIMEOUT = 10
RATE_LIMIT_DELAY = 0.5  # seconds between requests

# Preprocessing
REMOVE_STOPWORDS = False
LEMMATIZE = False

# Severity Prediction
SEVERITY_MODEL_TYPE = 'ensemble'  # 'svm', 'rf', 'bert', or 'ensemble'

# Logging
LOG_LEVEL = 'INFO'

print(f"Configuration loaded. Using device: {DEVICE}")
if DATA_DIR not in [DATABASE_URL]:
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
