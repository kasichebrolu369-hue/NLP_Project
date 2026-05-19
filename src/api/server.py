"""
FastAPI implementation for CVE NLP System
Provides REST API endpoints for data access and analysis
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import json
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")

# Create FastAPI app
app = FastAPI(
    title="CVE NLP Analysis API",
    description="API for analyzing and extracting information from CVE data using NLP",
    version="1.0.0"
)


# ==================== Pydantic Models ====================

class CVEResponse(BaseModel):
    """CVE response model"""
    cve_id: str
    description: str
    cvss_score: Optional[float] = None
    severity: Optional[str] = None
    published_date: Optional[str] = None
    cwe_ids: List[str] = []
    exploit_types: List[str] = []
    affected_os: List[str] = []
    affected_products: List[str] = []
    is_remote: bool = False
    requires_interaction: bool = False


class SeverityPredictionRequest(BaseModel):
    """Request for severity prediction"""
    description: str
    model_type: str = "bert"  # 'bert', 'svm', or 'rf'


class SeverityPredictionResponse(BaseModel):
    """Severity prediction response"""
    predicted_cvss: float
    severity_level: str
    confidence: float
    model_used: str


class TrendAnalysisResponse(BaseModel):
    """Trend analysis response"""
    exploit_type: str
    yearly_trends: Dict
    forecast: Optional[Dict] = None
    os_distribution: Dict


class DatabaseStatsResponse(BaseModel):
    """Database statistics"""
    total_cves: int
    severity_distribution: Dict[str, int]
    remote_exploitable: int
    average_cvss_score: Optional[float]


# ==================== Dependency Injection ====================

# These would be injected from the main application
cve_db = None
bert_extractor = None
severity_predictor = None
trend_analyzer = None


def set_dependencies(db, bert, severity, trend):
    """Set global dependencies"""
    global cve_db, bert_extractor, severity_predictor, trend_analyzer
    cve_db = db
    bert_extractor = bert
    severity_predictor = severity
    trend_analyzer = trend


# ==================== Health & Info Endpoints ====================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info", tags=["System"])
async def system_info():
    """Get system information"""
    if cve_db:
        stats = cve_db.get_statistics()
        return {
            "system": "CVE NLP Analysis System",
            "version": "1.0.0",
            "database_status": "connected",
            "statistics": stats
        }
    return {
        "system": "CVE NLP Analysis System",
        "version": "1.0.0",
        "database_status": "disconnected"
    }


# ==================== CVE Data Endpoints ====================

@app.get("/cve/{cve_id}", response_model=CVEResponse, tags=["CVE Data"])
async def get_cve(cve_id: str):
    """Get CVE by ID"""
    if not cve_db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    cve_data = cve_db.get_cve(cve_id)
    if not cve_data:
        raise HTTPException(status_code=404, detail=f"CVE {cve_id} not found")
    
    return CVEResponse(**cve_data)


@app.get("/cve/severity/{severity}", response_model=List[CVEResponse], tags=["CVE Data"])
async def get_cves_by_severity(
    severity: str = Query(..., regex="^(CRITICAL|HIGH|MEDIUM|LOW)$"),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get CVEs by severity level"""
    if not cve_db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    cves = cve_db.get_cves_by_severity(severity.upper(), limit=limit)
    return [CVEResponse(**cve) for cve in cves]


@app.get("/cve/exploit/{exploit_type}", response_model=List[CVEResponse], tags=["CVE Data"])
async def get_cves_by_exploit(
    exploit_type: str,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get CVEs by exploit type"""
    if not cve_db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    cves = cve_db.get_cves_by_exploit_type(exploit_type.upper(), limit=limit)
    return [CVEResponse(**cve) for cve in cves]


@app.get("/cve/os/{os_name}", response_model=List[CVEResponse], tags=["CVE Data"])
async def get_cves_by_os(
    os_name: str,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get CVEs affecting specific OS"""
    if not cve_db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    cves = cve_db.get_cves_by_os(os_name, limit=limit)
    return [CVEResponse(**cve) for cve in cves]


@app.get("/cve/date-range", response_model=List[CVEResponse], tags=["CVE Data"])
async def get_cves_by_date_range(
    start_date: str = Query(..., regex="^\\d{4}-\\d{2}-\\d{2}$"),
    end_date: str = Query(..., regex="^\\d{4}-\\d{2}-\\d{2}$"),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get CVEs published in date range"""
    if not cve_db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    cves = cve_db.get_cves_by_date_range(start, end, limit=limit)
    return [CVEResponse(**cve) for cve in cves]


# ==================== Analysis Endpoints ====================

@app.post("/analyze/severity", response_model=SeverityPredictionResponse, tags=["Analysis"])
async def predict_severity(request: SeverityPredictionRequest):
    """Predict CVSS severity from CVE description"""
    if not severity_predictor:
        raise HTTPException(status_code=503, detail="Severity predictor not initialized")
    
    try:
        # Make prediction
        description = request.description
        model_type = request.model_type.lower()
        
        if model_type == "bert":
            predicted_score = float(severity_predictor.predict([description])[0])
        else:
            predicted_score = float(severity_predictor.predict([description])[0])
        
        # Map to severity level
        if predicted_score >= 9.0:
            severity_level = "CRITICAL"
            confidence = min(predicted_score / 10, 1.0)
        elif predicted_score >= 7.0:
            severity_level = "HIGH"
            confidence = min(predicted_score / 10, 1.0)
        elif predicted_score >= 4.0:
            severity_level = "MEDIUM"
            confidence = min(predicted_score / 10, 1.0)
        else:
            severity_level = "LOW"
            confidence = min(predicted_score / 10, 1.0)
        
        return SeverityPredictionResponse(
            predicted_cvss=round(predicted_score, 2),
            severity_level=severity_level,
            confidence=round(confidence, 3),
            model_used=model_type
        )
    
    except Exception as e:
        logger.error(f"Error predicting severity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/trends", response_model=TrendAnalysisResponse, tags=["Analysis"])
async def analyze_trends(
    exploit_type: str = Query("RCE"),
    forecast_years: int = Query(3, ge=1, le=10)
):
    """Analyze vulnerability trends over time"""
    if not trend_analyzer:
        raise HTTPException(status_code=503, detail="Trend analyzer not initialized")
    
    try:
        # Get trends
        yearly_trends = trend_analyzer.analyze_exploit_trends(exploit_type)
        os_dist = trend_analyzer.analyze_os_distribution()
        forecast = trend_analyzer.forecast_trends(exploit_type, forecast_years)
        
        return TrendAnalysisResponse(
            exploit_type=exploit_type,
            yearly_trends=yearly_trends.to_dict('records'),
            forecast=forecast if forecast.get('forecast') is not None else None,
            os_distribution=os_dist.to_dict('records')
        )
    
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Statistics Endpoints ====================

@app.get("/stats", response_model=DatabaseStatsResponse, tags=["Statistics"])
async def get_database_stats():
    """Get database statistics"""
    if not cve_db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    stats = cve_db.get_statistics()
    return DatabaseStatsResponse(**stats)


# ==================== Search Endpoints ====================

@app.get("/search", response_model=List[CVEResponse], tags=["Search"])
async def search_cves(
    query: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=1000)
):
    """Search CVEs by keyword (searches descriptions and IDs)"""
    if not cve_db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    # This would require implementing a search function in the database
    # For now, return a placeholder
    raise HTTPException(status_code=501, detail="Search functionality not yet implemented")


# ==================== Root Endpoint ====================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with documentation"""
    return {
        "message": "CVE NLP Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "cve": "/cve/{cve_id}",
            "trends": "/analyze/trends",
            "severity": "/analyze/severity",
            "stats": "/stats"
        }
    }


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting CVE NLP API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
