"""
Main Pipeline Script
Orchestrates the entire CVE NLP analysis workflow
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.collector import CVEDataCollector
from preprocessing.processor import CVEDataProcessor
from extraction.extractor import CVEInformationExtractor
from database.models import DatabaseManager
from models.bert_extractor import BertCVEExtractor, BertCVETrainer, CVEBertDataset
from models.severity_predictor import ClassicalSeverityPredictor, TransformerSeverityPredictor
from analysis.trend_analyzer import TemporalTrendAnalyzer

from config.settings import (
    DATABASE_URL, BERT_MODEL, DEVICE, BATCH_SIZE, 
    LEARNING_RATE, EPOCHS, DATA_DIR, MODEL_DIR, RESULTS_DIR
)

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
from datetime import datetime
from loguru import logger
import json

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")


class CVENLPPipeline:
    """Main pipeline for CVE NLP analysis"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.collector = CVEDataCollector()
        self.processor = CVEDataProcessor(remove_stopwords=False, lemmatize=False)
        self.extractor = CVEInformationExtractor()
        self.db = DatabaseManager(DATABASE_URL)
        
        logger.info("CVE NLP Pipeline initialized")
    
    def stage_1_data_collection(self, limit: int = 100, keyword: str = None) -> pd.DataFrame:
        """
        Stage 1: Collect CVE data
        
        Args:
            limit: Number of CVEs to collect
            keyword: Optional keyword search
            
        Returns:
            DataFrame with raw CVE data
        """
        logger.info("="*50)
        logger.info("STAGE 1: Data Collection")
        logger.info("="*50)
        
        if keyword:
            logger.info(f"Searching CVEs with keyword: {keyword}")
            cves = self.collector.get_cves_by_keyword(keyword, limit=limit)
        else:
            logger.info("Fetching recent CVEs from NVD API...")
            cves = self.collector.get_cves_from_nvd(limit=limit)
        
        logger.info(f"Collected {len(cves)} CVEs")
        
        return cves
    
    def stage_2_preprocessing(self, cves: list) -> pd.DataFrame:
        """
        Stage 2: Preprocess CVE data
        
        Args:
            cves: List of raw CVE data
            
        Returns:
            DataFrame with preprocessed data
        """
        logger.info("="*50)
        logger.info("STAGE 2: Data Preprocessing")
        logger.info("="*50)
        
        df = self.processor.process_cve_list(cves)
        
        logger.info(f"Processed {len(df)} CVE records")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def stage_3_information_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 3: Extract structured information
        
        Args:
            df: Preprocessed CVE DataFrame
            
        Returns:
            DataFrame with extracted information
        """
        logger.info("="*50)
        logger.info("STAGE 3: Information Extraction (NER)")
        logger.info("="*50)
        
        extraction_results = []
        
        for idx, row in df.iterrows():
            logger.info(f"Extracting information from CVE {idx + 1}/{len(df)}: {row['cve_id']}")
            
            extracted = self.extractor.extract_cve_information(
                cve_id=row['cve_id'],
                description=row['description'],
                cvss_score=row.get('cvss_score')
            )
            
            # Merge with original row
            extraction_results.append({**row.to_dict(), **extracted})
        
        result_df = pd.DataFrame(extraction_results)
        logger.info(f"Extraction complete. {len(result_df)} CVEs processed")
        
        return result_df
    
    def stage_4_database_storage(self, df: pd.DataFrame) -> int:
        """
        Stage 4: Store extracted data in database
        
        Args:
            df: DataFrame with extracted CVE information
            
        Returns:
            Number of stored CVEs
        """
        logger.info("="*50)
        logger.info("STAGE 4: Database Storage")
        logger.info("="*50)
        
        count = self.db.add_cves_batch(df.to_dict('records'))
        
        logger.info(f"Stored {count} CVE records in database")
        
        return count
    
    def task_1_bert_fine_tuning(self, df: pd.DataFrame, epochs: int = 3) -> dict:
        """
        Task 1: Fine-tune BERT for structured information extraction
        
        Args:
            df: CVE DataFrame with labels
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        logger.info("="*50)
        logger.info("TASK 1: BERT Fine-tuning for Information Extraction")
        logger.info("="*50)
        
        device = DEVICE if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Prepare data
        texts = df['description'].tolist()
        
        # Create synthetic labels (in real scenario, these would be manually labeled)
        labels = []
        for _, row in df.iterrows():
            exploit_types = row.get('exploit_types', [])
            if isinstance(exploit_types, str):
                exploit_types = exploit_types.split(',') if exploit_types else []
            
            exploit_type = exploit_types[0] if exploit_types and len(exploit_types) > 0 else 'UNKNOWN'
            
            label = {
                'cwe': row.get('extracted_cwes', 'OTHER').split('|')[0] or 'OTHER',
                'exploit_type': exploit_type
            }
            labels.append(label)
        
        # Create dataset
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        dataset = CVEBertDataset(texts[:min(10, len(texts))], labels[:min(10, len(labels))], tokenizer)
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Initialize and train model
        model = BertCVEExtractor(
            bert_model=BERT_MODEL,
            num_cwe_classes=len(set(l['cwe'] for l in labels)),
            num_exploit_classes=len(set(l['exploit_type'] for l in labels))
        )
        trainer = BertCVETrainer(model, device=device)
        
        history = trainer.train(train_loader, val_loader, epochs=epochs, learning_rate=LEARNING_RATE)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, 'bert_cve_extractor.pt')
        os.makedirs(MODEL_DIR, exist_ok=True)
        trainer.save_model(model_path)
        
        logger.info(f"BERT model saved to {model_path}")
        
        return {
            'task': 'BERT Fine-tuning',
            'model_path': model_path,
            'history': history,
            'training_samples': len(train_loader),
            'validation_samples': len(val_loader)
        }
    
    def task_2_severity_prediction(self, df: pd.DataFrame) -> dict:
        """
        Task 2: Train and compare severity prediction models
        
        Args:
            df: CVE DataFrame with CVSS scores
            
        Returns:
            Comparison results
        """
        logger.info("="*50)
        logger.info("TASK 2: Severity/CVSS Score Prediction")
        logger.info("="*50)
        
        # Filter CVEs with CVSS scores
        df_with_scores = df[df['cvss_score'].notna()].copy()
        
        if len(df_with_scores) < 5:
            logger.warning("Insufficient CVEs with CVSS scores for training")
            return {'error': 'Insufficient data'}
        
        texts = df_with_scores['description'].tolist()
        scores = df_with_scores['cvss_score'].tolist()
        
        # Train Classical ML models
        logger.info("Training Classical ML models...")
        
        svm_model = ClassicalSeverityPredictor('svm')
        svm_metrics = svm_model.train(texts, scores)
        
        rf_model = ClassicalSeverityPredictor('rf')
        rf_metrics = rf_model.train(texts, scores)
        
        # Train Transformer model (if enough samples)
        device = DEVICE if torch.cuda.is_available() else 'cpu'
        logger.info("Training Transformer model...")
        
        transformer_model = TransformerSeverityPredictor(device=device)
        transformer_result = transformer_model.train(texts, scores, epochs=EPOCHS)
        
        results = {
            'task': 'Severity Prediction',
            'models': {
                'SVM': {
                    'MAE': svm_metrics['mae'],
                    'RMSE': svm_metrics['rmse'],
                    'R2': svm_metrics['r2']
                },
                'RandomForest': {
                    'MAE': rf_metrics['mae'],
                    'RMSE': rf_metrics['rmse'],
                    'R2': rf_metrics['r2']
                },
                'BERT': {
                    'MAE': transformer_result['final_metrics']['mae'],
                    'RMSE': transformer_result['final_metrics']['rmse'],
                    'R2': transformer_result['final_metrics']['r2']
                }
            },
            'training_samples': len(texts)
        }
        
        logger.info("Model Comparison Results:")
        for model_name, metrics in results['models'].items():
            logger.info(f"{model_name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")
        
        return results
    
    def task_3_temporal_analysis(self, df: pd.DataFrame) -> dict:
        """
        Task 3: Temporal trend analysis
        
        Args:
            df: CVE DataFrame with temporal information
            
        Returns:
            Analysis results
        """
        logger.info("="*50)
        logger.info("TASK 3: Temporal Trend Analysis")
        logger.info("="*50)
        
        analyzer = TemporalTrendAnalyzer(df)
        
        # Generate comprehensive report
        report = analyzer.generate_report()
        
        logger.info("Temporal Analysis Complete")
        
        return report
    
    def run_full_pipeline(self, num_cves: int = 50, keyword: str = None):
        """
        Run the complete pipeline
        
        Args:
            num_cves: Number of CVEs to process
            keyword: Optional keyword search
        """
        logger.info("\n" + "="*60)
        logger.info("CVE NLP ANALYSIS PIPELINE")
        logger.info("="*60 + "\n")
        
        start_time = datetime.now()
        
        try:
            # Stage 1: Data Collection
            raw_cves = self.stage_1_data_collection(limit=num_cves, keyword=keyword)
            
            # Stage 2: Preprocessing
            processed_df = self.stage_2_preprocessing(raw_cves)
            
            # Stage 3: Information Extraction
            extracted_df = self.stage_3_information_extraction(processed_df)
            
            # Stage 4: Database Storage
            self.stage_4_database_storage(extracted_df)
            
            # Task 1: BERT Fine-tuning
            task_1_results = self.task_1_bert_fine_tuning(extracted_df, epochs=2)
            
            # Task 2: Severity Prediction
            task_2_results = self.task_2_severity_prediction(extracted_df)
            
            # Task 3: Temporal Analysis
            task_3_results = self.task_3_temporal_analysis(extracted_df)
            
            # Save results
            results = {
                'timestamp': start_time.isoformat(),
                'total_cves_processed': len(extracted_df),
                'task_1_bert': task_1_results,
                'task_2_severity': task_2_results,
                'task_3_temporal': task_3_results
            }
            
            os.makedirs(RESULTS_DIR, exist_ok=True)
            results_path = os.path.join(RESULTS_DIR, f'pipeline_results_{start_time.strftime("%Y%m%d_%H%M%S")}.json')
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_path}")
            
            # Print summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE EXECUTION SUMMARY")
            logger.info("="*60)
            logger.info(f"Total CVEs Processed: {len(extracted_df)}")
            logger.info(f"Execution Time: {duration:.2f} seconds")
            logger.info(f"Results Saved: {results_path}")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CVE NLP Analysis Pipeline")
    parser.add_argument("--limit", type=int, default=20, help="Number of CVEs to process")
    parser.add_argument("--keyword", type=str, default=None, help="Keyword search")
    parser.add_argument("--collect-only", action="store_true", help="Only run data collection")
    
    args = parser.parse_args()
    
    pipeline = CVENLPPipeline()
    pipeline.run_full_pipeline(num_cves=args.limit, keyword=args.keyword)
