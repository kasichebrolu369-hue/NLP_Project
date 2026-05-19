"""
Data Preprocessing Module
Handles cleaning, tokenization, and normalization of CVE text
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from typing import List, Dict, Tuple
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Handles text preprocessing and cleaning"""
    
    def __init__(self, remove_stopwords: bool = False, lemmatize: bool = False):
        """
        Initialize preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # CVE-specific patterns
        self.cve_pattern = re.compile(r'CVE-\d{4}-\d{4,}')
        self.cwe_pattern = re.compile(r'CWE-\d+')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs but keep a placeholder
        text = self.url_pattern.sub('[URL]', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract important entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            'cve_ids': self.cve_pattern.findall(text),
            'cwe_ids': self.cwe_pattern.findall(text),
            'urls': self.url_pattern.findall(text)
        }
        return entities
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize if enabled
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into sentences
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def preprocess_cve_description(self, description: str) -> Dict:
        """
        Comprehensive preprocessing for CVE descriptions
        
        Args:
            description: Raw CVE description
            
        Returns:
            Dictionary with preprocessed data
        """
        # Extract entities first (before cleaning)
        entities = self.extract_entities(description)
        
        # Clean text
        cleaned = self.clean_text(description)
        
        # Tokenize into sentences
        sentences = self.sentence_tokenize(cleaned)
        
        # Tokenize into words
        tokens = self.tokenize(cleaned)
        
        return {
            'original': description,
            'cleaned': cleaned,
            'sentences': sentences,
            'tokens': tokens,
            'entities': entities,
            'length': len(cleaned),
            'token_count': len(tokens),
            'sentence_count': len(sentences)
        }


class CVEDataProcessor:
    """Processes CVE data collections"""
    
    def __init__(self, remove_stopwords: bool = False, lemmatize: bool = False):
        """
        Initialize CVE data processor
        
        Args:
            remove_stopwords: Remove stopwords from text
            lemmatize: Apply lemmatization
        """
        self.preprocessor = TextPreprocessor(remove_stopwords, lemmatize)
    
    def process_cve_list(self, cves: List[Dict]) -> pd.DataFrame:
        """
        Process a list of CVE entries
        
        Args:
            cves: List of CVE dictionaries from API
            
        Returns:
            Pandas DataFrame with processed CVE data
        """
        processed_data = []
        
        for idx, cve in enumerate(cves, 1):
            try:
                # Extract basic information
                cve_id = cve.get('id', '')
                descriptions = cve.get('descriptions', [])
                
                # Get English description (preferred) or first available
                description = next(
                    (d['value'] for d in descriptions if d.get('lang') == 'en'),
                    descriptions[0].get('value') if descriptions else ''
                )
                
                # Get CVSS score
                metrics = cve.get('metrics', {})
                cvss_score = None
                severity = None
                
                if 'cvssV3' in metrics or 'cvssV3_1' in metrics:
                    cvss_data = metrics.get('cvssV3') or metrics.get('cvssV3_1', [])[0]
                    if cvss_data:
                        cvss_score = cvss_data.get('cvssData', {}).get('baseScore')
                        severity = cvss_data.get('cvssData', {}).get('baseSeverity')
                elif 'cvssV2' in metrics:
                    cvss_data = metrics.get('cvssV2', [])[0]
                    if cvss_data:
                        cvss_score = cvss_data.get('cvssData', {}).get('baseScore')
                
                # Get published date
                published_date = cve.get('published', '')
                last_modified = cve.get('lastModified', '')
                
                # Get references
                references = cve.get('references', [])
                ref_urls = [ref.get('url', '') for ref in references]
                
                # Get weaknesses (CWEs)
                cwe_ids = []
                weaknesses = cve.get('weaknesses', [])
                for weakness in weaknesses:
                    cwe_list = weakness.get('description', [])
                    for cwe in cwe_list:
                        cwe_id = cwe.get('value', '')
                        if cwe_id:
                            cwe_ids.append(cwe_id)
                
                # Preprocess description
                preprocessed = self.preprocessor.preprocess_cve_description(description)
                
                # Create processed entry
                processed_entry = {
                    'cve_id': cve_id,
                    'description': description,
                    'cleaned_description': preprocessed['cleaned'],
                    'tokens': ' '.join(preprocessed['tokens']),  # For storage
                    'token_count': preprocessed['token_count'],
                    'sentence_count': preprocessed['sentence_count'],
                    'cvss_score': cvss_score,
                    'severity': severity,
                    'published_date': published_date,
                    'last_modified': last_modified,
                    'cwe_ids': '|'.join(cwe_ids),
                    'reference_count': len(ref_urls),
                    'extracted_cves': '|'.join(preprocessed['entities']['cve_ids']),
                    'extracted_cwes': '|'.join(preprocessed['entities']['cwe_ids']),
                }
                
                processed_data.append(processed_entry)
                
                if idx % 10 == 0:
                    logger.info(f"Processed {idx} CVEs...")
                    
            except Exception as e:
                logger.error(f"Error processing CVE {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_data)} CVEs")
        return pd.DataFrame(processed_data)
    
    def process_cve_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of CVEs
        
        Args:
            df: Input DataFrame with CVE data
            
        Returns:
            Processed DataFrame
        """
        df_copy = df.copy()
        
        # Preprocess description column
        logger.info("Preprocessing descriptions...")
        
        descriptions = []
        for desc in df_copy['description']:
            processed = self.preprocessor.preprocess_cve_description(str(desc))
            descriptions.append(processed)
        
        # Extract processed values
        df_copy['cleaned_description'] = [d['cleaned'] for d in descriptions]
        df_copy['tokens'] = [' '.join(d['tokens']) for d in descriptions]
        df_copy['token_count'] = [d['token_count'] for d in descriptions]
        df_copy['sentence_count'] = [d['sentence_count'] for d in descriptions]
        
        # Handle missing values
        df_copy = df_copy.fillna('')
        
        return df_copy


if __name__ == "__main__":
    # Example usage
    processor = CVEDataProcessor(remove_stopwords=False, lemmatize=False)
    
    # Sample CVE data
    sample_cves = [
        {
            'id': 'CVE-2024-1234',
            'descriptions': [
                {'lang': 'en', 'value': 'A remote code execution vulnerability in Application X allows attackers to execute arbitrary code...'}
            ],
            'metrics': {
                'cvssV3_1': [{'cvssData': {'baseScore': 9.8, 'baseSeverity': 'CRITICAL'}}]
            },
            'published': '2024-01-15',
            'lastModified': '2024-01-20',
            'references': [{'url': 'https://example.com'}],
            'weaknesses': [{'description': [{'value': 'CWE-119'}]}]
        }
    ]
    
    df = processor.process_cve_list(sample_cves)
    print(df.to_string())
