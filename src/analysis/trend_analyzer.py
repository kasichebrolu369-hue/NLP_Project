"""
Task 3: Temporal Trend Analysis
Analyzes RCE attacks over years and OS-based vulnerability distribution
Uses time-series modeling and trend analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")

warnings.filterwarnings('ignore')


class TemporalTrendAnalyzer:
    """Analyzes temporal trends in CVE data"""
    
    def __init__(self, cves_df: pd.DataFrame):
        """
        Initialize trend analyzer
        
        Args:
            cves_df: DataFrame with CVE data (must have 'published_date', 'exploit_types', 'affected_os' columns)
        """
        self.cves_df = cves_df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for analysis"""
        # Ensure published_date is datetime
        self.cves_df['published_date'] = pd.to_datetime(self.cves_df['published_date'], errors='coerce')
        
        # Extract year and month
        self.cves_df['year'] = self.cves_df['published_date'].dt.year
        self.cves_df['month'] = self.cves_df['published_date'].dt.month
        self.cves_df['year_month'] = self.cves_df['published_date'].dt.to_period('M')
        
        logger.info("Data preparation complete")
    
    def analyze_exploit_trends(self, exploit_type: str = 'RCE') -> pd.DataFrame:
        """
        Analyze trends for specific exploit type over years
        
        Args:
            exploit_type: Type of exploit to analyze (e.g., 'RCE')
            
        Returns:
            DataFrame with yearly statistics
        """
        logger.info(f"Analyzing {exploit_type} trends over years...")
        
        # Parse exploit types (assuming they're JSON arrays stored as strings)
        if isinstance(self.cves_df['exploit_types'].iloc[0], str):
            self.cves_df['exploit_list'] = self.cves_df['exploit_types'].apply(
                lambda x: eval(x) if x and x != '[]' else []
            )
        else:
            self.cves_df['exploit_list'] = self.cves_df['exploit_types']
        
        # Find CVEs with specific exploit type
        has_exploit = self.cves_df['exploit_list'].apply(
            lambda x: exploit_type in x if isinstance(x, list) else False
        )
        
        exploit_cves = self.cves_df[has_exploit]
        
        # Group by year
        yearly_trends = exploit_cves.groupby('year').size()
        
        # Calculate growth rate
        growth_rates = yearly_trends.pct_change() * 100
        
        # Create summary DataFrame
        trend_df = pd.DataFrame({
            'year': yearly_trends.index,
            f'{exploit_type}_count': yearly_trends.values,
            'growth_rate_%': growth_rates.values
        })
        
        logger.info(f"\n{exploit_type} Trend Summary:")
        logger.info(trend_df.to_string())
        
        return trend_df
    
    def analyze_os_distribution(self) -> pd.DataFrame:
        """
        Analyze vulnerability distribution across operating systems
        
        Returns:
            DataFrame with OS vulnerability statistics
        """
        logger.info("Analyzing OS-based vulnerability distribution...")
        
        # Parse affected_os (assuming they're JSON arrays)
        if isinstance(self.cves_df['affected_os'].iloc[0], str):
            self.cves_df['os_list'] = self.cves_df['affected_os'].apply(
                lambda x: eval(x) if x and x != '[]' else []
            )
        else:
            self.cves_df['os_list'] = self.cves_df['affected_os']
        
        # Count CVEs per OS
        os_counts = {}
        for os_list in self.cves_df['os_list']:
            if isinstance(os_list, list):
                for os_name in os_list:
                    os_counts[os_name] = os_counts.get(os_name, 0) + 1
        
        # Also count by severity
        os_severity = {}
        for idx, row in self.cves_df.iterrows():
            os_list = row.get('os_list', [])
            severity = row.get('severity', 'UNKNOWN')
            
            for os_name in os_list:
                if os_name not in os_severity:
                    os_severity[os_name] = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
                os_severity[os_name][severity] = os_severity[os_name].get(severity, 0) + 1
        
        # Create summary DataFrame
        if os_counts:
            os_df = pd.DataFrame([
                {
                    'OS': os_name,
                    'Total_CVEs': os_counts.get(os_name, 0),
                    **os_severity.get(os_name, {})
                }
                for os_name in os_counts.keys()
            ]).sort_values('Total_CVEs', ascending=False)
        else:
            # Return empty DataFrame with proper structure
            os_df = pd.DataFrame(columns=['OS', 'Total_CVEs', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'])
        
        logger.info("\nOS Vulnerability Distribution:")
        logger.info(os_df.to_string())
        
        return os_df
    
    def analyze_severity_trends(self) -> pd.DataFrame:
        """
        Analyze severity trends over time
        
        Returns:
            DataFrame with severity trends by year
        """
        logger.info("Analyzing severity trends over time...")
        
        # Group by year and severity
        severity_trends = self.cves_df.groupby(['year', 'severity']).size().unstack(fill_value=0)
        
        logger.info("\nSeverity Trends by Year:")
        logger.info(severity_trends)
        
        return severity_trends
    
    def forecast_trends(self, exploit_type: str = 'RCE', forecast_years: int = 3) -> Dict:
        """
        Forecast future trends using time series modeling
        
        Args:
            exploit_type: Type of exploit to forecast
            forecast_years: Number of years to forecast
            
        Returns:
            Dictionary with forecast results
        """
        logger.info(f"Forecasting {exploit_type} trends for next {forecast_years} years...")
        
        # Get historical data
        if isinstance(self.cves_df['exploit_types'].iloc[0], str):
            self.cves_df['exploit_list'] = self.cves_df['exploit_types'].apply(
                lambda x: eval(x) if x and x != '[]' else []
            )
        
        has_exploit = self.cves_df['exploit_list'].apply(
            lambda x: exploit_type in x if isinstance(x, list) else False
        )
        
        yearly_data = self.cves_df[has_exploit].groupby('year').size()
        
        if len(yearly_data) < 3:
            logger.warning(f"Insufficient historical data for forecasting ({len(yearly_data)} years)")
            return {'forecast': None, 'model': 'insufficient_data'}
        
        try:
            # Try ARIMA model
            model = ARIMA(yearly_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=forecast_years)
            
            logger.info(f"{exploit_type} Forecast:")
            for year, value in enumerate(forecast, start=max(yearly_data.index)+1):
                logger.info(f"  {year}: {value:.0f} CVEs")
            
            return {
                'forecast': forecast.values,
                'model': 'ARIMA',
                'historical': yearly_data.values,
                'years': list(range(yearly_data.index.min(), yearly_data.index.max() + forecast_years + 1))
            }
        
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            return {'forecast': None, 'error': str(e)}
    
    def monthly_trend_analysis(self, exploit_type: str = 'RCE') -> pd.Series:
        """
        Analyze monthly trends within a year
        
        Args:
            exploit_type: Type of exploit to analyze
            
        Returns:
            Series with monthly statistics
        """
        logger.info(f"Analyzing monthly trends for {exploit_type}...")
        
        if isinstance(self.cves_df['exploit_types'].iloc[0], str):
            self.cves_df['exploit_list'] = self.cves_df['exploit_types'].apply(
                lambda x: eval(x) if x and x != '[]' else []
            )
        
        has_exploit = self.cves_df['exploit_list'].apply(
            lambda x: exploit_type in x if isinstance(x, list) else False
        )
        
        monthly_data = self.cves_df[has_exploit].groupby('month').size()
        
        logger.info(f"\nMonthly Distribution of {exploit_type}:")
        logger.info(monthly_data)
        
        return monthly_data
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive trend analysis report
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Generating comprehensive trend analysis report...")
        
        report = {
            'metadata': {
                'total_cves': len(self.cves_df),
                'date_range': {
                    'start': self.cves_df['published_date'].min().isoformat() if 'published_date' in self.cves_df.columns else None,
                    'end': self.cves_df['published_date'].max().isoformat() if 'published_date' in self.cves_df.columns else None
                }
            },
            'exploit_trends': self.analyze_exploit_trends('RCE').to_dict('records'),
            'os_distribution': self.analyze_os_distribution().to_dict('records'),
            'severity_trends': self.analyze_severity_trends().to_dict(),
            'rce_forecast': self.forecast_trends('RCE', forecast_years=3),
            'monthly_trends': self.monthly_trend_analysis('RCE').to_dict()
        }
        
        logger.info("Report generation complete")
        
        return report
    
    def plot_trends(self, save_path: Optional[str] = None):
        """
        Create visualization of trends
        
        Args:
            save_path: Path to save the plot (optional)
        """
        # Prepare data
        if isinstance(self.cves_df['exploit_types'].iloc[0], str):
            self.cves_df['exploit_list'] = self.cves_df['exploit_types'].apply(
                lambda x: eval(x) if x and x != '[]' else []
            )
        
        has_rce = self.cves_df['exploit_list'].apply(
            lambda x: 'RCE' in x if isinstance(x, list) else False
        )
        
        yearly_rce = self.cves_df[has_rce].groupby('year').size()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RCE trend
        ax1 = axes[0, 0]
        yearly_rce.plot(ax=ax1, marker='o', color='red')
        ax1.set_title('RCE Vulnerabilities Over Time')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of CVEs')
        ax1.grid(True, alpha=0.3)
        
        # Severity distribution
        ax2 = axes[0, 1]
        severity_counts = self.cves_df['severity'].value_counts()
        severity_counts.plot(kind='bar', ax=ax2, color=['red', 'orange', 'yellow', 'green'])
        ax2.set_title('CVE Severity Distribution')
        ax2.set_xlabel('Severity')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # OS distribution
        ax3 = axes[1, 0]
        if isinstance(self.cves_df['affected_os'].iloc[0], str):
            self.cves_df['os_list'] = self.cves_df['affected_os'].apply(
                lambda x: eval(x) if x and x != '[]' else []
            )
        
        os_counts = {}
        for os_list in self.cves_df['os_list']:
            if isinstance(os_list, list):
                for os_name in os_list:
                    os_counts[os_name] = os_counts.get(os_name, 0) + 1
        
        if os_counts:
            os_df = pd.Series(os_counts).sort_values(ascending=False).head(10)
            os_df.plot(kind='barh', ax=ax3)
            ax3.set_title('Top affected Operating Systems')
            ax3.set_xlabel('Number of CVEs')
        
        # Monthly distribution
        ax4 = axes[1, 1]
        monthly_counts = self.cves_df.groupby('month').size()
        monthly_counts.plot(ax=ax4, marker='o', color='blue')
        ax4.set_title('Vulnerabilities by Month')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of CVEs')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage with sample data
    sample_data = {
        'cve_id': ['CVE-2024-1', 'CVE-2024-2', 'CVE-2023-1', 'CVE-2023-2'],
        'published_date': ['2024-01-15', '2024-06-20', '2023-03-10', '2023-11-05'],
        'exploit_types': ["['RCE']", "['XSS']", "['RCE']", "['RCE']"],
        'affected_os': ["['Windows', 'Linux']", "['Web']", "['Linux']", "['Windows']"],
        'severity': ['CRITICAL', 'MEDIUM', 'HIGH', 'CRITICAL']
    }
    
    df = pd.DataFrame(sample_data)
    
    analyzer = TemporalTrendAnalyzer(df)
    report = analyzer.generate_report()
    
    import json
    print(json.dumps(report, indent=2, default=str))
