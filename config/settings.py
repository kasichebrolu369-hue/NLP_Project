# CVE NLP Project Configuration

# Data Collection
CVE_BASE_URL = "https://www.cve.org"
CVE_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = "YOUR_NVD_API_KEY_HERE"
SCRAPE_TIMEOUT = 30

# Database
DATABASE_URL = "sqlite:///cve_database.db"
# DATABASE_URL = "postgresql://user:password@localhost/cve_db"

# Model Configuration
BERT_MODEL = "bert-base-uncased"
DEVICE = "cuda"  # or "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_SEQ_LENGTH = 512

# NER Configuration
NER_MODEL = "en_core_web_sm"
ENTITY_LABELS = ["CVE_ID", "CWE", "SEVERITY", "DATE", "DESCRIPTION"]

# Classification Categories
EXPLOIT_TYPES = ["RCE", "XSS", "SQL_INJECTION", "BUFFER_OVERFLOW", "PRIVILEGE_ESCALATION", "INFORMATION_DISCLOSURE", "DENIAL_OF_SERVICE"]
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# File Paths
DATA_DIR = "./data"
MODEL_DIR = "./models"
RESULTS_DIR = "./results"
LOG_DIR = "./logs"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time} | {level: <8} | {name}:{function}:{line} - {message}"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# Advanced Analysis
TREND_ANALYSIS_WINDOW = 30  # days
MIN_SEVERITY_SCORE = 5.0  # CVSS
