"""
Comprehensive Validation Reports Generator
Phase 3: ML Model Validation & Performance Analytics

Features:
- Detailed validation reports with statistical analysis
- Quantifiable improvements with confidence intervals
- Performance comparison dashboards
- Executive summaries for stakeholders
- Technical documentation for developers
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
import asyncio
import threading
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ReportType(Enum):
    """Types of validation reports"""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_DETAILED = "technical_detailed"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    STATISTICAL_VALIDATION = "statistical_validation"
    COMPARATIVE_STUDY = "comparative_study"
    RECOMMENDATIONS = "recommendations"


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    report_type: ReportType
    intersection_id: str
    generation_time: datetime
    validation_period: Tuple[datetime, datetime]
    summary_metrics: Dict[str, float]
    detailed_results: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    visualizations: List[str]  # File paths to generated charts
    recommendations: List[str]
    confidence_level: float
    report_file_path: str


@dataclass
class PerformanceImprovement:
    """Performance improvement metrics"""
    metric_name: str
    baseline_value: float
    optimized_value: float
    improvement_percentage: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    is_significant: bool
    practical_significance: str  # 'high', 'medium', 'low'


class StatisticalAnalyzer:
    """Statistical analysis for validation reports"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Statistical parameters
        self.alpha = self.config.get('alpha', 0.05)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.min_sample_size = self.config.get('min_sample_size', 30)
    
    def analyze_improvements(self, baseline_data: List[float], 
                           optimized_data: List[float]) -> PerformanceImprovement:
        """Analyze performance improvements with statistical significance"""
        try:
            # Basic statistics
            baseline_mean = np.mean(baseline_data)
            optimized_mean = np.mean(optimized_data)
            
            # Calculate improvement percentage
            improvement_percentage = ((optimized_mean - baseline_mean) / baseline_mean) * 100
            
            # Statistical tests
            t_stat, p_value = stats.ttest_ind(optimized_data, baseline_data)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(optimized_data) - 1) * np.var(optimized_data, ddof=1) + 
                                (len(baseline_data) - 1) * np.var(baseline_data, ddof=1)) / 
                               (len(optimized_data) + len(baseline_data) - 2))
            effect_size = (optimized_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for improvement
            se_diff = pooled_std * np.sqrt(1/len(optimized_data) + 1/len(baseline_data))
            t_critical = stats.t.ppf(1 - self.alpha/2, len(optimized_data) + len(baseline_data) - 2)
            margin_error = t_critical * se_diff
            ci_lower = (optimized_mean - baseline_mean) - margin_error
            ci_upper = (optimized_mean - baseline_mean) + margin_error
            
            # Determine significance
            is_significant = p_value < self.alpha
            
            # Determine practical significance
            if abs(effect_size) > 0.8:
                practical_significance = 'high'
            elif abs(effect_size) > 0.5:
                practical_significance = 'medium'
            else:
                practical_significance = 'low'
            
            return PerformanceImprovement(
                metric_name='performance_metric',
                baseline_value=baseline_mean,
                optimized_value=optimized_mean,
                improvement_percentage=improvement_percentage,
                confidence_interval=(ci_lower, ci_upper),
                p_value=p_value,
                effect_size=effect_size,
                is_significant=is_significant,
                practical_significance=practical_significance
            )
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}")
            return None
    
    def calculate_confidence_intervals(self, data: List[float], 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data"""
        try:
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            n = len(data)
            
            # Calculate standard error
            se = std / np.sqrt(n)
            
            # Calculate t-critical value
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, n - 1)
            
            # Calculate margin of error
            margin_error = t_critical * se
            
            return (mean - margin_error, mean + margin_error)
            
        except Exception:
            return (0.0, 0.0)
    
    def perform_anova_analysis(self, groups: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform ANOVA analysis for multiple groups"""
        try:
            group_names = list(groups.keys())
            group_data = [groups[name] for name in group_names]
            
            # Perform one-way ANOVA
            f_stat, p_value = stats.f_oneway(*group_data)
            
            # Calculate effect size (eta squared)
            all_data = np.concatenate(group_data)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in group_data)
            
            # Total sum of squares
            ss_total = sum((x - grand_mean)**2 for x in all_data)
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'is_significant': p_value < self.alpha,
                'group_means': {name: np.mean(data) for name, data in groups.items()},
                'group_stds': {name: np.std(data, ddof=1) for name, data in groups.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error in ANOVA analysis: {e}")
            return {}


class VisualizationGenerator:
    """Generate visualizations for validation reports"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Visualization settings
        self.output_dir = self.config.get('output_dir', 'reports/visualizations')
        self.figure_size = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 300)
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def create_performance_comparison_chart(self, baseline_data: List[float], 
                                          optimized_data: List[float],
                                          metric_name: str,
                                          intersection_id: str) -> str:
        """Create performance comparison chart"""
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
            
            # Box plot comparison
            data_to_plot = [baseline_data, optimized_data]
            labels = ['Baseline (Webster)', 'ML Optimized']
            
            box_plot = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightcoral')
            box_plot['boxes'][1].set_facecolor('lightblue')
            
            ax1.set_title(f'{metric_name} Comparison - {intersection_id}')
            ax1.set_ylabel(metric_name)
            ax1.grid(True, alpha=0.3)
            
            # Time series comparison
            time_points = range(len(baseline_data))
            ax2.plot(time_points, baseline_data, label='Baseline', color='red', alpha=0.7)
            ax2.plot(time_points, optimized_data, label='ML Optimized', color='blue', alpha=0.7)
            ax2.fill_between(time_points, baseline_data, optimized_data, alpha=0.2, color='green')
            
            ax2.set_title(f'{metric_name} Over Time - {intersection_id}')
            ax2.set_xlabel('Time Points')
            ax2.set_ylabel(metric_name)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Save figure
            filename = f"{intersection_id}_{metric_name}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = Path(self.output_dir) / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error creating performance comparison chart: {e}")
            return ""
    
    def create_improvement_dashboard(self, improvements: List[PerformanceImprovement],
                                   intersection_id: str) -> str:
        """Create improvement dashboard"""
        try:
            # Prepare data
            metrics = [imp.metric_name for imp in improvements]
            improvements_pct = [imp.improvement_percentage for imp in improvements]
            p_values = [imp.p_value for imp in improvements]
            effect_sizes = [imp.effect_size for imp in improvements]
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Improvement Percentage', 'Statistical Significance', 
                              'Effect Sizes', 'Summary Statistics'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "table"}]]
            )
            
            # Improvement percentages
            colors = ['green' if imp > 0 else 'red' for imp in improvements_pct]
            fig.add_trace(
                go.Bar(x=metrics, y=improvements_pct, name='Improvement %', 
                      marker_color=colors),
                row=1, col=1
            )
            
            # Statistical significance
            sig_colors = ['green' if p < 0.05 else 'red' for p in p_values]
            fig.add_trace(
                go.Bar(x=metrics, y=[-np.log10(p) for p in p_values], 
                      name='-log10(p-value)', marker_color=sig_colors),
                row=1, col=2
            )
            
            # Effect sizes
            effect_colors = ['green' if abs(e) > 0.5 else 'orange' if abs(e) > 0.2 else 'red' 
                           for e in effect_sizes]
            fig.add_trace(
                go.Bar(x=metrics, y=effect_sizes, name='Effect Size', 
                      marker_color=effect_colors),
                row=2, col=1
            )
            
            # Summary table
            summary_data = []
            for i, imp in enumerate(improvements):
                summary_data.append([
                    imp.metric_name,
                    f"{imp.improvement_percentage:.1f}%",
                    f"{imp.p_value:.3f}",
                    imp.practical_significance,
                    "Yes" if imp.is_significant else "No"
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Improvement %', 'P-value', 
                                      'Practical Significance', 'Significant']),
                    cells=dict(values=list(zip(*summary_data)) if summary_data else [[], [], [], [], []])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f"Performance Improvement Dashboard - {intersection_id}",
                height=800,
                showlegend=True
            )
            
            # Save figure
            filename = f"{intersection_id}_improvement_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = Path(self.output_dir) / filename
            fig.write_html(str(filepath))
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error creating improvement dashboard: {e}")
            return ""
    
    def create_learning_curve_visualization(self, learning_data: Dict[str, List[float]],
                                          intersection_id: str) -> str:
        """Create learning curve visualization"""
        try:
            fig = go.Figure()
            
            for algorithm, curve in learning_data.items():
                fig.add_trace(go.Scatter(
                    x=list(range(len(curve))),
                    y=curve,
                    mode='lines+markers',
                    name=algorithm,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=f"Learning Curves - {intersection_id}",
                xaxis_title="Epochs",
                yaxis_title="Performance Metric",
                height=600,
                showlegend=True
            )
            
            # Save figure
            filename = f"{intersection_id}_learning_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = Path(self.output_dir) / filename
            fig.write_html(str(filepath))
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error creating learning curve visualization: {e}")
            return ""


class ReportGenerator:
    """Generate comprehensive validation reports"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.statistical_analyzer = StatisticalAnalyzer(config.get('statistical_analyzer', {}))
        self.visualization_generator = VisualizationGenerator(config.get('visualization', {}))
        
        # Report settings
        self.output_dir = self.config.get('output_dir', 'reports')
        self.template_dir = self.config.get('template_dir', 'templates')
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Report templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load report templates"""
        templates = {}
        
        # Executive summary template
        executive_template = """
# Executive Summary - ML Traffic Optimization Validation

## Overview
This report presents the validation results for ML-based traffic optimization at {{ intersection_id }}.

## Key Findings
{% for improvement in improvements %}
- **{{ improvement.metric_name }}**: {{ "%.1f"|format(improvement.improvement_percentage) }}% improvement
  - Statistical Significance: {{ "Yes" if improvement.is_significant else "No" }}
  - Practical Significance: {{ improvement.practical_significance }}
{% endfor %}

## Recommendations
{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}

## Confidence Level
{{ "%.1f"|format(confidence_level * 100) }}%

Generated on: {{ generation_time }}
        """
        
        templates['executive'] = Template(executive_template)
        
        # Technical detailed template
        technical_template = """
# Technical Validation Report - {{ intersection_id }}

## Statistical Analysis
{% for improvement in improvements %}
### {{ improvement.metric_name }}
- Baseline Value: {{ "%.3f"|format(improvement.baseline_value) }}
- Optimized Value: {{ "%.3f"|format(improvement.optimized_value) }}
- Improvement: {{ "%.1f"|format(improvement.improvement_percentage) }}%
- Confidence Interval: ({{ "%.3f"|format(improvement.confidence_interval[0]) }}, {{ "%.3f"|format(improvement.confidence_interval[1]) }})
- P-value: {{ "%.6f"|format(improvement.p_value) }}
- Effect Size (Cohen's d): {{ "%.3f"|format(improvement.effect_size) }}
- Significant: {{ "Yes" if improvement.is_significant else "No" }}
{% endfor %}

## Visualizations
{% for viz in visualizations %}
- {{ viz }}
{% endfor %}

## Detailed Results
{{ detailed_results | tojson(indent=2) }}
        """
        
        templates['technical'] = Template(technical_template)
        
        return templates
    
    def generate_validation_report(self, intersection_id: str,
                                 baseline_data: Dict[str, List[float]],
                                 optimized_data: Dict[str, List[float]],
                                 validation_period: Tuple[datetime, datetime],
                                 additional_metrics: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """Generate comprehensive validation report"""
        try:
            report_id = str(uuid.uuid4())
            generation_time = datetime.now()
            
            # Analyze improvements for each metric
            improvements = []
            for metric_name in baseline_data.keys():
                if metric_name in optimized_data:
                    improvement = self.statistical_analyzer.analyze_improvements(
                        baseline_data[metric_name],
                        optimized_data[metric_name]
                    )
                    if improvement:
                        improvement.metric_name = metric_name
                        improvements.append(improvement)
            
            # Generate visualizations
            visualizations = []
            for metric_name in baseline_data.keys():
                if metric_name in optimized_data:
                    chart_path = self.visualization_generator.create_performance_comparison_chart(
                        baseline_data[metric_name],
                        optimized_data[metric_name],
                        metric_name,
                        intersection_id
                    )
                    if chart_path:
                        visualizations.append(chart_path)
            
            # Create improvement dashboard
            if improvements:
                dashboard_path = self.visualization_generator.create_improvement_dashboard(
                    improvements, intersection_id
                )
                if dashboard_path:
                    visualizations.append(dashboard_path)
            
            # Calculate summary metrics
            summary_metrics = self._calculate_summary_metrics(improvements)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(improvements, additional_metrics)
            
            # Calculate overall confidence level
            confidence_level = self._calculate_overall_confidence(improvements)
            
            # Create detailed results
            detailed_results = {
                'improvements': [asdict(imp) for imp in improvements],
                'statistical_tests': self._perform_additional_tests(baseline_data, optimized_data),
                'validation_metadata': {
                    'intersection_id': intersection_id,
                    'validation_period': validation_period,
                    'baseline_sample_size': sum(len(data) for data in baseline_data.values()),
                    'optimized_sample_size': sum(len(data) for data in optimized_data.values()),
                    'metrics_tested': list(baseline_data.keys())
                }
            }
            
            # Generate report files
            report_files = self._generate_report_files(
                intersection_id, improvements, recommendations, 
                confidence_level, generation_time, visualizations, detailed_results
            )
            
            # Create validation report
            report = ValidationReport(
                report_id=report_id,
                report_type=ReportType.TECHNICAL_DETAILED,
                intersection_id=intersection_id,
                generation_time=generation_time,
                validation_period=validation_period,
                summary_metrics=summary_metrics,
                detailed_results=detailed_results,
                statistical_analysis=self._perform_statistical_analysis(improvements),
                visualizations=visualizations,
                recommendations=recommendations,
                confidence_level=confidence_level,
                report_file_path=report_files.get('main_report', '')
            )
            
            self.logger.info(f"Generated validation report for {intersection_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {e}")
            return None
    
    def _calculate_summary_metrics(self, improvements: List[PerformanceImprovement]) -> Dict[str, float]:
        """Calculate summary metrics from improvements"""
        if not improvements:
            return {}
        
        return {
            'average_improvement': np.mean([imp.improvement_percentage for imp in improvements]),
            'median_improvement': np.median([imp.improvement_percentage for imp in improvements]),
            'max_improvement': max([imp.improvement_percentage for imp in improvements]),
            'min_improvement': min([imp.improvement_percentage for imp in improvements]),
            'significant_improvements': sum([1 for imp in improvements if imp.is_significant]),
            'total_metrics': len(improvements),
            'high_practical_significance': sum([1 for imp in improvements if imp.practical_significance == 'high']),
            'average_effect_size': np.mean([imp.effect_size for imp in improvements])
        }
    
    def _generate_recommendations(self, improvements: List[PerformanceImprovement], 
                                additional_metrics: Optional[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not improvements:
            recommendations.append("Insufficient data for meaningful analysis")
            return recommendations
        
        # Analyze improvement patterns
        significant_improvements = [imp for imp in improvements if imp.is_significant]
        high_significance = [imp for imp in improvements if imp.practical_significance == 'high']
        
        if len(significant_improvements) >= len(improvements) * 0.8:
            recommendations.append("ML optimization shows strong statistical significance across most metrics")
        elif len(significant_improvements) >= len(improvements) * 0.5:
            recommendations.append("ML optimization shows moderate statistical significance")
        else:
            recommendations.append("ML optimization shows limited statistical significance - consider model refinement")
        
        if len(high_significance) >= len(improvements) * 0.6:
            recommendations.append("ML optimization demonstrates high practical significance")
        elif len(high_significance) >= len(improvements) * 0.3:
            recommendations.append("ML optimization shows moderate practical significance")
        else:
            recommendations.append("ML optimization shows limited practical significance - review implementation")
        
        # Specific metric recommendations
        for imp in improvements:
            if imp.improvement_percentage > 30:
                recommendations.append(f"Excellent improvement in {imp.metric_name} - consider scaling to other intersections")
            elif imp.improvement_percentage > 15:
                recommendations.append(f"Good improvement in {imp.metric_name} - monitor for consistency")
            elif imp.improvement_percentage < 5:
                recommendations.append(f"Limited improvement in {imp.metric_name} - investigate optimization opportunities")
        
        # General recommendations
        avg_improvement = np.mean([imp.improvement_percentage for imp in improvements])
        if avg_improvement > 25:
            recommendations.append("Overall performance exceeds target - ready for production deployment")
        elif avg_improvement > 15:
            recommendations.append("Overall performance meets target - proceed with controlled deployment")
        else:
            recommendations.append("Overall performance below target - additional optimization required")
        
        return recommendations
    
    def _calculate_overall_confidence(self, improvements: List[PerformanceImprovement]) -> float:
        """Calculate overall confidence level"""
        if not improvements:
            return 0.0
        
        # Calculate confidence based on statistical significance and effect sizes
        significant_count = sum([1 for imp in improvements if imp.is_significant])
        high_effect_count = sum([1 for imp in improvements if abs(imp.effect_size) > 0.5])
        
        significance_confidence = significant_count / len(improvements)
        effect_confidence = high_effect_count / len(improvements)
        
        # Weighted average
        overall_confidence = (significance_confidence * 0.6) + (effect_confidence * 0.4)
        
        return min(1.0, overall_confidence)
    
    def _perform_additional_tests(self, baseline_data: Dict[str, List[float]], 
                                optimized_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform additional statistical tests"""
        tests = {}
        
        # ANOVA test for multiple metrics
        all_baseline = []
        all_optimized = []
        
        for metric_name in baseline_data.keys():
            if metric_name in optimized_data:
                all_baseline.extend(baseline_data[metric_name])
                all_optimized.extend(optimized_data[metric_name])
        
        if all_baseline and all_optimized:
            groups = {
                'baseline': all_baseline,
                'optimized': all_optimized
            }
            anova_result = self.statistical_analyzer.perform_anova_analysis(groups)
            tests['anova'] = anova_result
        
        return tests
    
    def _perform_statistical_analysis(self, improvements: List[PerformanceImprovement]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        if not improvements:
            return {}
        
        analysis = {
            'total_metrics': len(improvements),
            'significant_metrics': sum([1 for imp in improvements if imp.is_significant]),
            'high_practical_significance': sum([1 for imp in improvements if imp.practical_significance == 'high']),
            'improvement_distribution': {
                'mean': np.mean([imp.improvement_percentage for imp in improvements]),
                'std': np.std([imp.improvement_percentage for imp in improvements]),
                'min': min([imp.improvement_percentage for imp in improvements]),
                'max': max([imp.improvement_percentage for imp in improvements]),
                'median': np.median([imp.improvement_percentage for imp in improvements])
            },
            'effect_size_distribution': {
                'mean': np.mean([imp.effect_size for imp in improvements]),
                'std': np.std([imp.effect_size for imp in improvements]),
                'min': min([imp.effect_size for imp in improvements]),
                'max': max([imp.effect_size for imp in improvements])
            }
        }
        
        return analysis
    
    def _generate_report_files(self, intersection_id: str, improvements: List[PerformanceImprovement],
                             recommendations: List[str], confidence_level: float,
                             generation_time: datetime, visualizations: List[str],
                             detailed_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate report files"""
        report_files = {}
        
        try:
            # Generate executive summary
            executive_content = self.templates['executive'].render(
                intersection_id=intersection_id,
                improvements=improvements,
                recommendations=recommendations,
                confidence_level=confidence_level,
                generation_time=generation_time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            executive_file = Path(self.output_dir) / f"{intersection_id}_executive_summary_{generation_time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(executive_file, 'w') as f:
                f.write(executive_content)
            report_files['executive'] = str(executive_file)
            
            # Generate technical report
            technical_content = self.templates['technical'].render(
                intersection_id=intersection_id,
                improvements=improvements,
                visualizations=visualizations,
                detailed_results=detailed_results
            )
            
            technical_file = Path(self.output_dir) / f"{intersection_id}_technical_report_{generation_time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(technical_file, 'w') as f:
                f.write(technical_content)
            report_files['technical'] = str(technical_file)
            
            # Generate JSON report
            json_report = {
                'intersection_id': intersection_id,
                'generation_time': generation_time.isoformat(),
                'improvements': [asdict(imp) for imp in improvements],
                'recommendations': recommendations,
                'confidence_level': confidence_level,
                'visualizations': visualizations,
                'summary_metrics': self._calculate_summary_metrics(improvements)
            }
            
            json_file = Path(self.output_dir) / f"{intersection_id}_validation_report_{generation_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_file, 'w') as f:
                json.dump(json_report, f, indent=2)
            report_files['json'] = str(json_file)
            
            report_files['main_report'] = str(technical_file)
            
        except Exception as e:
            self.logger.error(f"Error generating report files: {e}")
        
        return report_files
    
    def generate_comprehensive_report(self, validation_data: Dict[str, Dict[str, Any]]) -> Dict[str, ValidationReport]:
        """Generate comprehensive validation report for multiple intersections"""
        reports = {}
        
        for intersection_id, data in validation_data.items():
            try:
                report = self.generate_validation_report(
                    intersection_id=intersection_id,
                    baseline_data=data.get('baseline_data', {}),
                    optimized_data=data.get('optimized_data', {}),
                    validation_period=data.get('validation_period', (datetime.now() - timedelta(days=1), datetime.now())),
                    additional_metrics=data.get('additional_metrics', {})
                )
                
                if report:
                    reports[intersection_id] = report
                    
            except Exception as e:
                self.logger.error(f"Error generating report for {intersection_id}: {e}")
        
        return reports
    
    def generate_summary_report(self, reports: Dict[str, ValidationReport]) -> str:
        """Generate summary report across all intersections"""
        try:
            # Calculate overall statistics
            all_improvements = []
            for report in reports.values():
                if 'improvements' in report.detailed_results:
                    for imp_data in report.detailed_results['improvements']:
                        all_improvements.append(imp_data['improvement_percentage'])
            
            if not all_improvements:
                return "No improvement data available for summary report"
            
            # Generate summary statistics
            summary_stats = {
                'total_intersections': len(reports),
                'total_metrics': len(all_improvements),
                'average_improvement': np.mean(all_improvements),
                'median_improvement': np.median(all_improvements),
                'std_improvement': np.std(all_improvements),
                'min_improvement': min(all_improvements),
                'max_improvement': max(all_improvements),
                'intersections_above_target': sum([1 for imp in all_improvements if imp > 30]),
                'intersections_meeting_target': sum([1 for imp in all_improvements if 15 <= imp <= 30]),
                'intersections_below_target': sum([1 for imp in all_improvements if imp < 15])
            }
            
            # Generate summary content
            summary_content = f"""
# Comprehensive Validation Summary Report

## Overall Performance
- **Total Intersections**: {summary_stats['total_intersections']}
- **Total Metrics Analyzed**: {summary_stats['total_metrics']}
- **Average Improvement**: {summary_stats['average_improvement']:.1f}%
- **Median Improvement**: {summary_stats['median_improvement']:.1f}%
- **Standard Deviation**: {summary_stats['std_improvement']:.1f}%

## Target Achievement
- **Above Target (30%+)**: {summary_stats['intersections_above_target']} metrics
- **Meeting Target (15-30%)**: {summary_stats['intersections_meeting_target']} metrics
- **Below Target (<15%)**: {summary_stats['intersections_below_target']} metrics

## Per-Intersection Results
"""
            
            for intersection_id, report in reports.items():
                summary_metrics = report.summary_metrics
                summary_content += f"""
### {intersection_id}
- Average Improvement: {summary_metrics.get('average_improvement', 0):.1f}%
- Significant Metrics: {summary_metrics.get('significant_improvements', 0)}/{summary_metrics.get('total_metrics', 0)}
- Confidence Level: {report.confidence_level:.1%}
"""
            
            # Save summary report
            summary_file = Path(self.output_dir) / f"comprehensive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(summary_file, 'w') as f:
                f.write(summary_content)
            
            return str(summary_file)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return ""


class ValidationReports:
    """
    Main Validation Reports System
    
    Features:
    - Detailed validation reports with statistical analysis
    - Quantifiable improvements with confidence intervals
    - Performance comparison dashboards
    - Executive summaries for stakeholders
    - Technical documentation for developers
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.report_generator = ReportGenerator(config.get('report_generator', {}))
        
        # Report storage
        self.generated_reports = {}
        self.report_history = deque(maxlen=1000)
        
        self.logger.info("Validation Reports system initialized")
    
    def generate_validation_report(self, intersection_id: str,
                                 baseline_data: Dict[str, List[float]],
                                 optimized_data: Dict[str, List[float]],
                                 validation_period: Tuple[datetime, datetime],
                                 additional_metrics: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """Generate validation report for single intersection"""
        try:
            report = self.report_generator.generate_validation_report(
                intersection_id, baseline_data, optimized_data, 
                validation_period, additional_metrics
            )
            
            if report:
                self.generated_reports[intersection_id] = report
                self.report_history.append(report)
                
                self.logger.info(f"Generated validation report for {intersection_id}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {e}")
            return None
    
    def generate_comprehensive_reports(self, validation_data: Dict[str, Dict[str, Any]]) -> Dict[str, ValidationReport]:
        """Generate comprehensive validation reports for multiple intersections"""
        try:
            reports = self.report_generator.generate_comprehensive_report(validation_data)
            
            # Generate summary report
            summary_file = self.report_generator.generate_summary_report(reports)
            
            self.logger.info(f"Generated comprehensive reports for {len(reports)} intersections")
            
            return reports
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive reports: {e}")
            return {}
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary of generated reports"""
        return {
            'total_reports': len(self.generated_reports),
            'report_history_size': len(self.report_history),
            'intersections_with_reports': list(self.generated_reports.keys()),
            'latest_report_time': max([r.generation_time for r in self.report_history]) if self.report_history else None
        }
    
    def get_intersection_report(self, intersection_id: str) -> Optional[ValidationReport]:
        """Get validation report for specific intersection"""
        return self.generated_reports.get(intersection_id)
    
    def export_reports(self, output_format: str = 'json') -> Dict[str, str]:
        """Export all reports in specified format"""
        try:
            export_files = {}
            
            for intersection_id, report in self.generated_reports.items():
                if output_format == 'json':
                    export_data = {
                        'report_id': report.report_id,
                        'intersection_id': report.intersection_id,
                        'generation_time': report.generation_time.isoformat(),
                        'summary_metrics': report.summary_metrics,
                        'recommendations': report.recommendations,
                        'confidence_level': report.confidence_level
                    }
                    
                    filename = f"{intersection_id}_report_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    filepath = Path(self.config.get('output_dir', 'reports')) / filename
                    
                    with open(filepath, 'w') as f:
                        json.dump(export_data, f, indent=2)
                    
                    export_files[intersection_id] = str(filepath)
            
            return export_files
            
        except Exception as e:
            self.logger.error(f"Error exporting reports: {e}")
            return {}
