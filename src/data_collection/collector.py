"""
CVE Data Collection Module
Handles scraping and API-based data collection from CVE sources
"""

import requests
import json
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")


class CVEDataCollector:
    """Collects CVE data from multiple sources"""
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize CVE Data Collector
        
        Args:
            api_key: NVD API key for authenticated requests
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.nvd_base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        self.cve_base_url = "https://www.cve.org"
        self.session = requests.Session()
        self.rate_limit_delay = 6  # NVD API rate limit: 1 request per 6 seconds
        self.last_request_time = 0
        
    def _wait_for_rate_limit(self):
        """Respect API rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_cves_from_nvd(self, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None,
                         limit: int = 100) -> List[Dict]:
        """
        Fetch CVEs from NVD API
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of CVEs to fetch
            
        Returns:
            List of CVE dictionaries
        """
        logger.info(f"Fetching CVEs from NVD API (limit: {limit})")
        
        cves = []
        start_index = 0
        results_per_page = min(limit, 2000)  # NVD API max per page
        
        params = {
            "resultsPerPage": results_per_page,
            "startIndex": start_index
        }
        
        if start_date:
            params["pubStartDate"] = f"{start_date}T00:00:00Z"
        if end_date:
            params["pubEndDate"] = f"{end_date}T23:59:59Z"
        
        if self.api_key:
            params["apiKey"] = self.api_key
        
        try:
            while start_index < limit:
                params["startIndex"] = start_index
                
                self._wait_for_rate_limit()
                response = self.session.get(self.nvd_base_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                vulnerabilities = data.get("vulnerabilities", [])
                
                if not vulnerabilities:
                    logger.info(f"No more CVEs found. Total fetched: {len(cves)}")
                    break
                
                for vuln in vulnerabilities:
                    cve_data = vuln.get("cve", {})
                    cves.append(cve_data)
                
                start_index += results_per_page
                logger.info(f"Fetched {len(cves)} CVEs so far...")
                
                if len(cves) >= limit:
                    break
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from NVD API: {e}")
        
        return cves[:limit]
    
    def scrape_cve_details(self, cve_id: str) -> Optional[Dict]:
        """
        Scrape detailed CVE information from CVE.org
        
        Args:
            cve_id: CVE identifier (e.g., CVE-2024-1234)
            
        Returns:
            Dictionary containing CVE details or None if not found
        """
        try:
            url = f"{self.cve_base_url}/CVERecord?id={cve_id}"
            self._wait_for_rate_limit()
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            cve_details = {
                'cve_id': cve_id,
                'url': url,
                'title': soup.find('h2') and soup.find('h2').get_text() or '',
                'description': '',
                'references': [],
                'scrape_time': datetime.now().isoformat()
            }
            
            # Extract description
            desc_section = soup.find('div', {'id': 'description'})
            if desc_section:
                cve_details['description'] = desc_section.get_text(strip=True)
            
            # Extract references
            refs = soup.find_all('a', {'class': 'reference'})
            cve_details['references'] = [ref.get('href', '') for ref in refs]
            
            logger.info(f"Successfully scraped details for {cve_id}")
            return cve_details
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping CVE {cve_id}: {e}")
            return None
    
    def get_cves_by_keyword(self, keyword: str, limit: int = 50) -> List[Dict]:
        """
        Search CVEs by keyword using NVD API
        
        Args:
            keyword: Search keyword
            limit: Maximum results
            
        Returns:
            List of matching CVEs
        """
        logger.info(f"Searching CVEs with keyword: {keyword}")
        
        params = {
            "keywordSearch": keyword,
            "resultsPerPage": min(limit, 2000),
            "startIndex": 0
        }
        
        if self.api_key:
            params["apiKey"] = self.api_key
        
        try:
            self._wait_for_rate_limit()
            response = self.session.get(self.nvd_base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            vulnerabilities = data.get("vulnerabilities", [])
            
            cves = [vuln.get("cve", {}) for vuln in vulnerabilities]
            logger.info(f"Found {len(cves)} CVEs matching keyword")
            return cves[:limit]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching CVEs: {e}")
            return []
    
    def batch_scrape_cves(self, cve_ids: List[str]) -> List[Dict]:
        """
        Scrape multiple CVEs
        
        Args:
            cve_ids: List of CVE identifiers
            
        Returns:
            List of CVE details
        """
        results = []
        for idx, cve_id in enumerate(cve_ids, 1):
            logger.info(f"Scraping CVE {idx}/{len(cve_ids)}: {cve_id}")
            details = self.scrape_cve_details(cve_id)
            if details:
                results.append(details)
        
        return results


if __name__ == "__main__":
    # Example usage
    collector = CVEDataCollector()
    
    # Fetch recent CVEs
    recent_cves = collector.get_cves_from_nvd(limit=10)
    print(f"Fetched {len(recent_cves)} recent CVEs")
    
    # Search by keyword
    rce_cves = collector.get_cves_by_keyword("remote code execution", limit=5)
    print(f"Found {len(rce_cves)} RCE-related CVEs")
