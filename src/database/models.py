"""
Database Module
Handles storage and retrieval of CVE data
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Optional
import json
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")

Base = declarative_base()


class CVERecord(Base):
    """Database model for CVE records"""
    
    __tablename__ = 'cve_records'
    
    cve_id = Column(String(50), primary_key=True)
    description = Column(Text)
    cleaned_description = Column(Text)
    tokens = Column(Text)
    cvss_score = Column(Float)
    severity = Column(String(20))
    published_date = Column(DateTime)
    last_modified = Column(DateTime)
    cwe_ids = Column(Text)
    reference_count = Column(Integer, default=0)
    extracted_cves = Column(Text)
    extracted_cwes = Column(Text)
    
    # Extracted information
    exploit_types = Column(Text)  # JSON array as string
    affected_os = Column(Text)    # JSON array as string
    affected_products = Column(Text)  # JSON array as string
    is_remote = Column(Integer, default=0)  # Boolean as integer
    requires_interaction = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert record to dictionary"""
        return {
            'cve_id': self.cve_id,
            'description': self.description,
            'cvss_score': self.cvss_score,
            'severity': self.severity,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'cwe_ids': self.cwe_ids.split('|') if self.cwe_ids else [],
            'exploit_types': json.loads(self.exploit_types) if self.exploit_types else [],
            'affected_os': json.loads(self.affected_os) if self.affected_os else [],
            'affected_products': json.loads(self.affected_products) if self.affected_products else [],
            'is_remote': bool(self.is_remote),
            'requires_interaction': bool(self.requires_interaction)
        }


class CEVEAnalysisRecord(Base):
    """Database model for CEVE analysis results"""
    
    __tablename__ = 'analysis_records'
    
    analysis_id = Column(String(100), primary_key=True)
    cve_id = Column(String(50))
    analysis_type = Column(String(50))  # 'bert_extraction', 'severity_prediction', 'trend'
    analysis_results = Column(Text)  # JSON
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert record to dictionary"""
        return {
            'analysis_id': self.analysis_id,
            'cve_id': self.cve_id,
            'analysis_type': self.analysis_type,
            'analysis_results': json.loads(self.analysis_results),
            'confidence_score': self.confidence_score
        }


class DatabaseManager:
    """Manages database operations"""
    
    def __init__(self, database_url: str = "sqlite:///cve_database.db"):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info(f"Initialized database: {database_url}")
    
    def add_cve(self, cve_data: dict, session: Optional[Session] = None) -> bool:
        """
        Add a CVE record to database
        
        Args:
            cve_data: Dictionary with CVE data
            session: Database session (creates new if not provided)
            
        Returns:
            True if successful
        """
        close_session = False
        if session is None:
            session = self.SessionLocal()
            close_session = True
        
        try:
            # Parse JSON fields
            exploit_types = cve_data.get('exploit_types', [])
            affected_os = cve_data.get('affected_os', [])
            affected_products = cve_data.get('affected_products', [])
            
            record = CVERecord(
                cve_id=cve_data.get('cve_id'),
                description=cve_data.get('description'),
                cleaned_description=cve_data.get('cleaned_description'),
                tokens=cve_data.get('tokens'),
                cvss_score=cve_data.get('cvss_score'),
                severity=cve_data.get('severity'),
                published_date=cve_data.get('published_date'),
                last_modified=cve_data.get('last_modified'),
                cwe_ids=cve_data.get('cwe_ids'),
                reference_count=cve_data.get('reference_count', 0),
                extracted_cves=cve_data.get('extracted_cves'),
                extracted_cwes=cve_data.get('extracted_cwes'),
                exploit_types=json.dumps(exploit_types),
                affected_os=json.dumps(affected_os),
                affected_products=json.dumps(affected_products),
                is_remote=int(cve_data.get('is_remote', False)),
                requires_interaction=int(cve_data.get('requires_interaction', False))
            )
            
            session.merge(record)  # Use merge to handle duplicates
            session.commit()
            logger.info(f"Added CVE record: {cve_data.get('cve_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding CVE record: {e}")
            session.rollback()
            return False
        
        finally:
            if close_session:
                session.close()
    
    def add_cves_batch(self, cves_data: List[dict]) -> int:
        """
        Add multiple CVE records
        
        Args:
            cves_data: List of CVE data dictionaries
            
        Returns:
            Number of successfully added records
        """
        session = self.SessionLocal()
        success_count = 0
        
        try:
            for idx, cve_data in enumerate(cves_data, 1):
                if self.add_cve(cve_data, session):
                    success_count += 1
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx}/{len(cves_data)} CVEs...")
            
            session.commit()
            logger.info(f"Successfully added {success_count}/{len(cves_data)} CVE records")
            
        finally:
            session.close()
        
        return success_count
    
    def get_cve(self, cve_id: str) -> Optional[dict]:
        """
        Retrieve a CVE record
        
        Args:
            cve_id: CVE identifier
            
        Returns:
            CVE data dictionary or None
        """
        session = self.SessionLocal()
        try:
            record = session.query(CVERecord).filter_by(cve_id=cve_id).first()
            return record.to_dict() if record else None
        finally:
            session.close()
    
    def get_cves_by_severity(self, severity: str, limit: int = 100) -> List[dict]:
        """
        Get CVEs by severity level
        
        Args:
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            limit: Maximum results
            
        Returns:
            List of CVE dictionaries
        """
        session = self.SessionLocal()
        try:
            records = session.query(CVERecord).filter_by(severity=severity).limit(limit).all()
            return [r.to_dict() for r in records]
        finally:
            session.close()
    
    def get_cves_by_exploit_type(self, exploit_type: str, limit: int = 100) -> List[dict]:
        """
        Get CVEs by exploit type
        
        Args:
            exploit_type: Exploit type (RCE, XSS, etc.)
            limit: Maximum results
            
        Returns:
            List of CVE dictionaries
        """
        session = self.SessionLocal()
        try:
            records = session.query(CVERecord).all()
            results = []
            
            for record in records:
                types = json.loads(record.exploit_types) if record.exploit_types else []
                if exploit_type in types:
                    results.append(record.to_dict())
                    if len(results) >= limit:
                        break
            
            return results
        finally:
            session.close()
    
    def get_cves_by_os(self, os_name: str, limit: int = 100) -> List[dict]:
        """
        Get CVEs affecting a specific OS
        
        Args:
            os_name: Operating system name
            limit: Maximum results
            
        Returns:
            List of CVE dictionaries
        """
        session = self.SessionLocal()
        try:
            records = session.query(CVERecord).all()
            results = []
            
            for record in records:
                os_list = json.loads(record.affected_os) if record.affected_os else []
                if os_name in os_list:
                    results.append(record.to_dict())
                    if len(results) >= limit:
                        break
            
            return results
        finally:
            session.close()
    
    def get_cves_by_date_range(self, start_date: datetime, end_date: datetime, 
                              limit: int = 100) -> List[dict]:
        """
        Get CVEs published in a date range
        
        Args:
            start_date: Start date
            end_date: End date
            limit: Maximum results
            
        Returns:
            List of CVE dictionaries
        """
        session = self.SessionLocal()
        try:
            records = session.query(CVERecord).filter(
                CVERecord.published_date >= start_date,
                CVERecord.published_date <= end_date
            ).limit(limit).all()
            
            return [r.to_dict() for r in records]
        finally:
            session.close()
    
    def get_statistics(self) -> dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        session = self.SessionLocal()
        try:
            total_cves = session.query(CVERecord).count()
            
            severity_counts = {}
            for severity_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = session.query(CVERecord).filter_by(severity=severity_level).count()
                severity_counts[severity_level] = count
            
            remote_exploitable = session.query(CVERecord).filter_by(is_remote=1).count()
            
            avg_cvss = None
            records = session.query(CVERecord).all()
            cvss_scores = [r.cvss_score for r in records if r.cvss_score]
            if cvss_scores:
                avg_cvss = sum(cvss_scores) / len(cvss_scores)
            
            return {
                'total_cves': total_cves,
                'severity_distribution': severity_counts,
                'remote_exploitable': remote_exploitable,
                'average_cvss_score': avg_cvss
            }
        
        finally:
            session.close()
    
    def add_analysis_result(self, analysis_data: dict) -> bool:
        """
        Add analysis result to database
        
        Args:
            analysis_data: Dictionary with analysis data
            
        Returns:
            True if successful
        """
        session = self.SessionLocal()
        
        try:
            record = CEVEAnalysisRecord(
                analysis_id=analysis_data.get('analysis_id'),
                cve_id=analysis_data.get('cve_id'),
                analysis_type=analysis_data.get('analysis_type'),
                analysis_results=json.dumps(analysis_data.get('results', {})),
                confidence_score=analysis_data.get('confidence_score', 0.0)
            )
            
            session.add(record)
            session.commit()
            logger.info(f"Added analysis result: {analysis_data.get('analysis_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding analysis result: {e}")
            session.rollback()
            return False
        
        finally:
            session.close()


if __name__ == "__main__":
    # Example usage
    db = DatabaseManager()
    
    # Test adding a CVE
    sample_cve = {
        'cve_id': 'CVE-2024-1234',
        'description': 'Sample vulnerability description',
        'cvss_score': 9.8,
        'severity': 'CRITICAL',
        'published_date': datetime.now(),
        'exploit_types': ['RCE'],
        'affected_os': ['Linux', 'Windows']
    }
    
    db.add_cve(sample_cve)
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Database statistics: {stats}")
