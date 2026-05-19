"""
Named Entity Recognition (NER) and Information Extraction Module
Extracts structured information from CVE descriptions
"""

import spacy
from typing import List, Dict, Tuple, Set
import re
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")


class CVEInformationExtractor:
    """Extracts structured information from CVE data using NER and pattern matching"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize information extractor
        
        Args:
            model_name: SpaCy model to use for NER
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded SpaCy model: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
        
        # Define CVE-specific patterns and keywords
        self.severity_keywords = {
            'critical': ['critical', 'critical severity', 'cvss 9', 'cvss 10'],
            'high': ['high', 'high severity', 'cvss 7', 'cvss 8'],
            'medium': ['medium', 'medium severity', 'cvss 5', 'cvss 6'],
            'low': ['low', 'low severity', 'cvss 4', 'cvss 3']
        }
        
        self.exploit_keywords = {
            'RCE': ['remote code execution', 'rce', 'arbitrary code execution', 'code execution', 'command execution'],
            'XSS': ['cross-site scripting', 'xss', 'script injection', 'stored xss', 'reflected xss'],
            'SQL_INJECTION': ['sql injection', 'sql', 'database injection', 'sqli'],
            'BUFFER_OVERFLOW': ['buffer overflow', 'buffer over-read', 'stack overflow', 'heap overflow'],
            'PRIVILEGE_ESCALATION': ['privilege escalation', 'privilege elevation', 'escalation', 'elevated privileges'],
            'INFORMATION_DISCLOSURE': ['information disclosure', 'information leak', 'data leak', 'sensitive information'],
            'DENIAL_OF_SERVICE': ['denial of service', 'dos', 'ddos', 'crash', 'hang', 'resource exhaustion']
        }
        
        self.os_keywords = {
            'Windows': ['windows', 'microsoft', 'win32', 'winnt', '.exe'],
            'Linux': ['linux', 'ubuntu', 'debian', 'centos', 'fedora', 'rhel'],
            'macOS': ['macos', 'mac os', 'osx', 'apple'],
            'Android': ['android', 'apk'],
            'iOS': ['ios', 'iphone', 'ipad'],
            'Web': ['web', 'browser', 'http', 'apache', 'nginx']
        }
    
    def _find_keywords(self, text: str, keyword_dict: Dict[str, List[str]]) -> Set[str]:
        """
        Find keywords in text (case-insensitive)
        
        Args:
            text: Text to search in
            keyword_dict: Dictionary mapping categories to keyword lists
            
        Returns:
            Set of found categories
        """
        text_lower = text.lower()
        found = set()
        
        for category, keywords in keyword_dict.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found.add(category)
                    break
        
        return found
    
    def extract_entities_ner(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Extract named entities from text using SpaCy NER
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'PRODUCT': [],
            'OTHER': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append((ent.text, ent.label_))
            else:
                entities['OTHER'].append((ent.text, ent.label_))
        
        return entities
    
    def extract_severity(self, text: str, cvss_score: float = None) -> str:
        """
        Extract severity level from text
        
        Args:
            text: CVE description text
            cvss_score: CVSS score if available
            
        Returns:
            Severity level (CRITICAL, HIGH, MEDIUM, LOW)
        """
        # First check CVSS score if available
        if cvss_score is not None:
            if cvss_score >= 9.0:
                return 'CRITICAL'
            elif cvss_score >= 7.0:
                return 'HIGH'
            elif cvss_score >= 4.0:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        # Otherwise use keyword-based approach
        found_severities = self._find_keywords(text, self.severity_keywords)
        
        if found_severities:
            severity_priority = ['critical', 'high', 'medium', 'low']
            for sev in severity_priority:
                if sev in found_severities:
                    return sev.upper()
        
        return 'UNKNOWN'
    
    def extract_exploit_types(self, text: str) -> List[str]:
        """
        Extract exploit types from text
        
        Args:
            text: CVE description text
            
        Returns:
            List of identified exploit types
        """
        return list(self._find_keywords(text, self.exploit_keywords))
    
    def extract_affected_os(self, text: str) -> List[str]:
        """
        Extract affected operating systems from text
        
        Args:
            text: CVE description text
            
        Returns:
            List of affected operating systems
        """
        return list(self._find_keywords(text, self.os_keywords))
    
    def extract_affected_products(self, text: str) -> List[str]:
        """
        Extract product names from text using NER
        
        Args:
            text: CVE description text
            
        Returns:
            List of potentially affected products
        """
        doc = self.nlp(text)
        products = []
        
        for ent in doc.ents:
            # ORG entities often represent product vendors
            if ent.label_ == 'ORG':
                products.append(ent.text)
        
        # Also look for common CVE product references
        product_pattern = re.compile(r'(?:^|\s)([A-Z][A-Za-z0-9\s\-\.]+)\s+(?:version|v\.|\d+\.\d+)', re.MULTILINE)
        matches = product_pattern.findall(text)
        products.extend(matches)
        
        return list(set(products))
    
    def extract_cve_information(self, cve_id: str, description: str, 
                               cvss_score: float = None) -> Dict:
        """
        Comprehensive information extraction from CVE entry
        
        Args:
            cve_id: CVE identifier
            description: CVE description text
            cvss_score: CVSS score if available
            
        Returns:
            Dictionary with extracted information
        """
        # Extract all information
        ner_entities = self.extract_entities_ner(description)
        severity = self.extract_severity(description, cvss_score)
        exploit_types = self.extract_exploit_types(description)
        affected_os = self.extract_affected_os(description)
        affected_products = self.extract_affected_products(description)
        
        # Parse dependency relationships
        doc = self.nlp(description)
        dependencies = self._extract_dependencies(doc)
        
        return {
            'cve_id': cve_id,
            'extracted_severity': severity,
            'exploit_types': exploit_types,
            'affected_os': affected_os,
            'affected_products': affected_products,
            'named_entities': ner_entities,
            'key_phrases': self._extract_key_phrases(doc),
            'dependencies': dependencies,
            'is_remote': self._check_remote_exploitability(description),
            'requires_interaction': self._check_user_interaction(description)
        }
    
    def _extract_dependencies(self, doc) -> List[Dict]:
        """
        Extract dependency relationships from text
        
        Args:
            doc: SpaCy parsed document
            
        Returns:
            List of dependency relationships
        """
        dependencies = []
        
        for token in doc:
            if token.dep_ in ['nsubj', 'obj', 'iobj']:
                dependencies.append({
                    'head': token.head.text,
                    'child': token.text,
                    'relation': token.dep_
                })
        
        return dependencies
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """
        Extract key noun phrases from text
        
        Args:
            doc: SpaCy parsed document
            
        Returns:
            List of key phrases
        """
        key_phrases = []
        
        for chunk in doc.noun_chunks:
            if len(chunk) > 1:  # Multi-word phrases
                key_phrases.append(chunk.text)
        
        return key_phrases[:10]  # Return top 10
    
    def _check_remote_exploitability(self, text: str) -> bool:
        """
        Check if vulnerability is remotely exploitable
        
        Args:
            text: CVE description
            
        Returns:
            True if vulnerability appears to be remotely exploitable
        """
        remote_keywords = ['remote', 'network', 'internet', 'http', 'ssl', 'https', 'tcp', 'udp']
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in remote_keywords)
    
    def _check_user_interaction(self, text: str) -> bool:
        """
        Check if exploitation requires user interaction
        
        Args:
            text: CVE description
            
        Returns:
            True if user interaction is required
        """
        interaction_keywords = ['user', 'click', 'visit', 'open', 'interact', 'action', 'interaction']
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in interaction_keywords)


if __name__ == "__main__":
    # Example usage
    extractor = CVEInformationExtractor()
    
    sample_description = """
    A remote code execution vulnerability exists in Application XYZ that allows 
    an attacker to execute arbitrary code on affected Windows and Linux systems 
    through a specially crafted network request. The vulnerability affects versions 
    prior to 2.0. This is a critical severity issue with CVSS score of 9.8.
    """
    
    extracted = extractor.extract_cve_information(
        "CVE-2024-1234",
        sample_description,
        cvss_score=9.8
    )
    
    import json
    print(json.dumps(extracted, indent=2, default=str))
