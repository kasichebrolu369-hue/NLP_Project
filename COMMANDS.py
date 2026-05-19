#!/usr/bin/env python
"""
EXACT COMMANDS TO RUN THE PROJECT
Copy and paste any of these commands to run different parts
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CVE NLP PROJECT - COMMAND REFERENCE                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

# ============================================================================
# STEP 1: SETUP (First Time Only)
# ============================================================================

# Copy the entire line and paste into PowerShell/Command Prompt:

pip install -r "b:\\Data Science Projects(Own)\\NLP_project\\requirements.txt"

# OR install critical packages separately:

pip install torch transformers nltk spacy sqlalchemy fastapi uvicorn pandas scikit-learn --no-deps && python -m spacy download en_core_web_sm


# ============================================================================
# OPTION A: RUN THE DEMO (What Just Ran)
# ============================================================================

cd "b:\\Data Science Projects(Own)\\NLP_project" && c:/python313/python.exe demo_run.py

  → Shows project overview and structure
  → Time: ~5 seconds
  → Output: Formatted project information


# ============================================================================
# OPTION B: RUN FULL PIPELINE (Main Program)
# ============================================================================

cd "b:\\Data Science Projects(Own)\\NLP_project" && c:/python313/python.exe src/main_pipeline.py --limit 5

  → Executes: Data collection → Preprocessing → Extraction → Storage → All 3 tasks
  → Time: 3-10 minutes (depending on API response)
  → Output: JSON report in results/ folder
  → Variations:
     - --limit 3   (Quick demo, ~2 minutes)
     - --limit 10  (Standard run, ~5 minutes)
     - --limit 100 (Full analysis, ~15 minutes)


# ============================================================================
# OPTION C: START JUPYTER NOTEBOOKS
# ============================================================================

cd "b:\\Data Science Projects(Own)\\NLP_project" && jupyter notebook

  → Opens Jupyter Lab
  → Time: ~30 seconds to load
  → Access: http://localhost:8888
  → Notebooks available:
     1. notebooks/07_complete_pipeline.ipynb  (Full workflow)
     2. notebooks/04_task1_bert.ipynb        (BERT fine-tuning)
     3. notebooks/05_task2_severity.ipynb    (Severity comparison)
     4. notebooks/06_task3_trends.ipynb      (Trend analysis)


# ============================================================================
# OPTION D: START REST API SERVER
# ============================================================================

cd "b:\\Data Science Projects(Own)\\NLP_project" && python src/api/server.py

  → Starts FastAPI server
  → Time: ~5 seconds to startup
  → Access: http://localhost:8000
  → Features:
     - Swagger UI: http://localhost:8000/docs
     - ReDoc: http://localhost:8000/redoc
     - 15+ endpoints available
  → Stop: Ctrl+C in terminal


# Example API calls (in PowerShell):

# Test health
Invoke-WebRequest -Uri "http://localhost:8000/health"

# Get database stats
Invoke-WebRequest -Uri "http://localhost:8000/stats"

# Predict severity (with JSON body)
$body = @{
    description = "Remote code execution vulnerability in Apache server"
    model_type = "bert"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/analyze/severity" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body


# ============================================================================
# OPTION E: QUICK VERIFICATION TESTS
# ============================================================================

# Test 1: Check all imports
c:/python313/python.exe -c "import torch, transformers, pandas, nltk, spacy; print('✓ All packages installed')"

# Test 2: Verify project structure
dir "b:\\Data Science Projects(Own)\\NLP_project\\src"

# Test 3: Check config file
type "b:\\Data Science Projects(Own)\\NLP_project\\config\\settings.py"

# Test 4: Count files
Get-ChildItem -Path "b:\\Data Science Projects(Own)\\NLP_project\\src" -Recurse -Include "*.py" | Measure-Object


# ============================================================================
# OPTION F: RUN INDIVIDUAL COMPONENTS IN PYTHON
# ============================================================================

# Start Python interactive shell
cd "b:\\Data Science Projects(Own)\\NLP_project" && python

# Copy-paste these lines one by one:

import sys
sys.path.insert(0, 'src')

# Test 1: Data Collection
from data_collection.collector import CVEDataCollector
collector = CVEDataCollector()
cves = collector.get_cves_from_nvd(limit=2)
print(f"Fetched {len(cves)} CVEs")

# Test 2: Preprocessing
from preprocessing.processor import CVEDataProcessor
processor = CVEDataProcessor()
processed = processor.process_cve_list(cves)
print(processed.head())

# Test 3: Information Extraction
from extraction.extractor import CVEInformationExtractor
extractor = CVEInformationExtractor()
extracted = extractor.extract_cve_information(cves[0]['id'], cves[0].get('description', ''))
print(extracted)

# Test 4: Trend Analysis (if data exists)
from analysis.trend_analyzer import TemporalTrendAnalyzer
import pandas as pd
df = processed
if len(df) > 0:
    analyzer = TemporalTrendAnalyzer(df)
    report = analyzer.generate_report()
    print(report)

# To exit Python: type 'exit()' and press Enter


# ============================================================================
# OPTION G: VIEW PROJECT DOCUMENTATION
# ============================================================================

# View README (comprehensive guide)
type "b:\\Data Science Projects(Own)\\NLP_project\\README.md" | more

# View Quick Start
type "b:\\Data Science Projects(Own)\\NLP_project\\QUICKSTART.md" | more

# View Configuration
type "b:\\Data Science Projects(Own)\\NLP_project\\config\\settings.py"


# ============================================================================
# OPTION H: ADVANCED - MODIFY AND RUN WITH CUSTOM SETTINGS
# ============================================================================

# Edit config file (in VS Code or any editor)
code "b:\\Data Science Projects(Own)\\NLP_project\\config\\settings.py"

# After editing, run with new settings
cd "b:\\Data Science Projects(Own)\\NLP_project" && python src/main_pipeline.py --limit 10

# Common modifications:
# - Change DEVICE = "cpu"  (if GPU issues)
# - Change BATCH_SIZE = 16 (if memory issues)
# - Change DATABASE_URL (if using PostgreSQL)
# - Change EPOCHS = 5 (for better training)


# ============================================================================
# TROUBLESHOOTING COMMANDS
# ============================================================================

# If you get import errors:
pip install --upgrade torch transformers --no-deps

# If you need to download spaCy model:
python -m spacy download en_core_web_sm

# If database is locked:
del "b:\\Data Science Projects(Own)\\NLP_project\\cve_database.db"

# If API port is already in use:
# Restart PowerShell or wait 60 seconds

# Check Python version:
python --version

# Check installed packages:
pip list | grep -E "torch|transformers|pandas|spacy"


# ============================================================================
# BATCH SCRIPT - RUN EVERYTHING IN SEQUENCE
# ============================================================================

# Save this as "run_all.bat" in the project folder:

@echo off
cd "b:\\Data Science Projects(Own)\\NLP_project"

echo ========================================
echo CVE NLP PROJECT - COMPLETE RUN
echo ========================================

echo.
echo [1/3] Running Demo...
c:/python313/python.exe demo_run.py

echo.
echo [2/3] Running Main Pipeline...
c:/python313/python.exe src/main_pipeline.py --limit 5

echo.
echo [3/3] Pipeline Complete!
echo Results saved to: results/
echo Database saved to: cve_database.db

echo.
echo ========================================
echo Done! To explore data:
echo   jupyter notebook
echo To start API:
echo   python src/api/server.py
echo ========================================


# ============================================================================
# RECOMMENDED WORKFLOW
# ============================================================================

# FOR FIRST-TIME USERS:
# 1. Run demo
#    c:/python313/python.exe demo_run.py
# 
# 2. Run small pipeline
#    python src/main_pipeline.py --limit 3
#
# 3. Explore in Jupyter
#    jupyter notebook
#
# 4. Deploy API (optional)
#    python src/api/server.py


# FOR ADVANCED USERS:
# 1. Edit config/settings.py (adjust hyperparameters)
#
# 2. Run full pipeline
#    python src/main_pipeline.py --limit 100
#
# 3. Start API server
#    python src/api/server.py
#
# 4. Integrate into your application (import pipeline)


# ============================================================================
# IMPORTANT NOTES
# ============================================================================

# - First run downloads BERT model (~420MB) - takes time
# - NVD API has rate limiting - add delays for large requests
# - GPU recommended but CPU works (slower)
# - Requires internet for data collection
# - SQLite database created on first run (~5MB per 100 CVEs)


# ============================================================================
# STILL NEED HELP?
# ============================================================================

# Check these files:
# - README.md (520 lines)
# - QUICKSTART.md (200 lines)
# - PROJECT_STATUS.md (complete overview)
# - RUN_PROJECT.md (execution guide)

# Or check module docstrings in Python:
# python -c "from src.main_pipeline import CVENLPPipeline; help(CVENLPPipeline)"

╚══════════════════════════════════════════════════════════════════════════════╝

Ready to run! Pick any command above and copy-paste it. 🚀
""")

# Display this file path for easy reference
import os
script_path = os.path.abspath(__file__)
print(f"\nThis file: {script_path}")
print("Save this output for quick reference!")
