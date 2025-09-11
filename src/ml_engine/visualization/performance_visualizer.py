"""
Performance Visualization Utilities for ML Traffic Signal Optimization
Comprehensive visualization tools for performance analysis and monitoring
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceVisualizer:
    """
    Comprehensive visualization system for traffic signal optimization performance
    
    Features:
    - Real-time performance dashboards
    - Historical trend analysis
    - Algorithm comparison charts
    - System health monitoring
    - A/B testing visualizations
    - Export capabilities for reports
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6A994E',
            'light': '#F2F2F2',
            'dark': '#2D3436'
        }
        
        # Algorithm colors
        self.algorithm_colors = {
            'q_learning': '#E74C3C',
            'dynamic_programming': '#3498DB',
            'websters_formula': '#2ECC71',
            'ensemble': '#9B59B6',
            'control': '#95A5A6'
        }
    
    def plot_performance_trends(self, performance_data: Dict[str, List[float]], 
                              timestamps: List[datetime] = None,
                              title: str = "Performance Trends",
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot performance trends over time"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Wait Time
        axes[0, 0].plot(timestamps or range(len(performance_data.get('wait_time', []))), 
                       performance_data.get('wait_time', []), 
                       color=self.colors['primary'], linewidth=2)
        axes[0, 0].set_title('Wait Time (seconds)', fontweight='bold')
        axes[0, 0].set_ylabel('Wait Time (s)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Throughput
        axes[0, 1].plot(timestamps or range(len(performance_data.get('throughput', []))), 
                       performance_data.get('throughput', []), 
                       color=self.colors['success'], linewidth=2)
        axes[0, 1].set_title('Throughput (vehicles/hour)', fontweight='bold')
        axes[0, 1].set_ylabel('Throughput (vph)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Efficiency
        axes[1, 0].plot(timestamps or range(len(performance_data.get('efficiency', []))), 
                       performance_data.get('efficiency', []), 
                       color=self.colors['warning'], linewidth=2)
        axes[1, 0].set_title('Efficiency Score', fontweight='bold')
        axes[1, 0].set_ylabel('Efficiency')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence
        axes[1, 1].plot(timestamps or range(len(performance_data.get('confidence', []))), 
                       performance_data.get('confidence', []), 
                       color=self.colors['info'], linewidth=2)
        axes[1, 1].set_title('Confidence Score', fontweight='bold')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            if timestamps:
                ax.tick_params(axis='x', rotation=45)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Performance trends plot saved to {save_path}")
        
        return fig
    
    def plot_algorithm_comparison(self, algorithm_data: Dict[str, Dict[str, List[float]]],
                                metrics: List[str] = None,
                                title: str = "Algorithm Performance Comparison",
                                save_path: Optional[str] = None) -> plt.Figure:
        """Compare performance across different algorithms"""
        
        if metrics is None:
            metrics = ['wait_time', 'throughput', 'efficiency', 'confidence']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Prepare data for box plot
            data_for_plot = []
            labels = []
            
            for algorithm, data in algorithm_data.items():
                if metric in data and data[metric]:
                    data_for_plot.append(data[metric])
                    labels.append(algorithm.replace('_', ' ').title())
            
            if data_for_plot:
                # Create box plot
                box_plot = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
                
                # Color the boxes
                for patch, algorithm in zip(box_plot['boxes'], algorithm_data.keys()):
                    patch.set_facecolor(self.algorithm_colors.get(algorithm, self.colors['primary']))
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Algorithm comparison plot saved to {save_path}")
        
        return fig
    
    def plot_heatmap_analysis(self, data: pd.DataFrame, 
                            x_col: str, y_col: str, value_col: str,
                            title: str = "Performance Heatmap",
                            save_path: Optional[str] = None) -> plt.Figure:
        """Create heatmap for performance analysis"""
        
        # Pivot data for heatmap
        pivot_data = data.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': value_col})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Heatmap saved to {save_path}")
        
        return fig
    
    def plot_ab_test_results(self, test_data: Dict[str, Any],
                           title: str = "A/B Test Results",
                           save_path: Optional[str] = None) -> plt.Figure:
        """Visualize A/B test results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        variants = test_data.get('variant_results', {})
        significance = test_data.get('statistical_significance', {})
        effect_sizes = test_data.get('effect_sizes', {})
        
        # Plot 1: Performance comparison
        ax1 = axes[0, 0]
        variant_names = list(variants.keys())
        metrics = ['wait_time', 'throughput', 'efficiency']
        
        x = np.arange(len(variant_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = []
            for variant in variant_names:
                if metric in variants[variant]:
                    values.append(variants[variant][metric]['mean'])
                else:
                    values.append(0)
            
            ax1.bar(x + i * width, values, width, 
                   label=metric.replace('_', ' ').title(),
                   color=list(self.colors.values())[i])
        
        ax1.set_xlabel('Variants')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([v.replace('_', ' ').title() for v in variant_names])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Statistical significance
        ax2 = axes[0, 1]
        sig_data = []
        sig_labels = []
        
        for key, sig in significance.items():
            sig_data.append(1 if 'significant' in sig.value else 0)
            sig_labels.append(key.replace('_', ' ').title())
        
        colors = [self.colors['success'] if x == 1 else self.colors['warning'] for x in sig_data]
        ax2.bar(sig_labels, sig_data, color=colors)
        ax2.set_title('Statistical Significance')
        ax2.set_ylabel('Significant (1) / Not Significant (0)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Effect sizes
        ax3 = axes[1, 0]
        effect_data = list(effect_sizes.values())
        effect_labels = [k.replace('_', ' ').title() for k in effect_sizes.keys()]
        
        colors = [self.colors['success'] if x > 0 else self.colors['warning'] for x in effect_data]
        ax3.bar(effect_labels, effect_data, color=colors)
        ax3.set_title('Effect Sizes')
        ax3.set_ylabel('Effect Size')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 4: Sample sizes
        ax4 = axes[1, 1]
        sample_sizes = []
        sample_labels = []
        
        for variant, data in variants.items():
            if 'wait_time' in data:
                sample_sizes.append(data['wait_time']['count'])
                sample_labels.append(variant.replace('_', ' ').title())
        
        ax4.bar(sample_labels, sample_sizes, color=self.colors['info'])
        ax4.set_title('Sample Sizes')
        ax4.set_ylabel('Number of Samples')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"A/B test results plot saved to {save_path}")
        
        return fig
    
    def plot_system_health(self, health_data: Dict[str, List[float]],
                          timestamps: List[datetime] = None,
                          title: str = "System Health Monitoring",
                          save_path: Optional[str] = None) -> plt.Figure:
        """Plot system health metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # CPU Usage
        axes[0, 0].plot(timestamps or range(len(health_data.get('cpu_usage', []))), 
                       health_data.get('cpu_usage', []), 
                       color=self.colors['primary'], linewidth=2)
        axes[0, 0].axhline(y=80, color=self.colors['warning'], linestyle='--', alpha=0.7, label='Warning (80%)')
        axes[0, 0].set_title('CPU Usage', fontweight='bold')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory Usage
        axes[0, 1].plot(timestamps or range(len(health_data.get('memory_usage', []))), 
                       health_data.get('memory_usage', []), 
                       color=self.colors['secondary'], linewidth=2)
        axes[0, 1].axhline(y=85, color=self.colors['warning'], linestyle='--', alpha=0.7, label='Warning (85%)')
        axes[0, 1].set_title('Memory Usage', fontweight='bold')
        axes[0, 1].set_ylabel('Memory Usage (%)')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error Rate
        axes[1, 0].plot(timestamps or range(len(health_data.get('error_rate', []))), 
                       health_data.get('error_rate', []), 
                       color=self.colors['warning'], linewidth=2)
        axes[1, 0].axhline(y=0.05, color=self.colors['warning'], linestyle='--', alpha=0.7, label='Warning (5%)')
        axes[1, 0].set_title('Error Rate', fontweight='bold')
        axes[1, 0].set_ylabel('Error Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Response Time
        axes[1, 1].plot(timestamps or range(len(health_data.get('avg_response_time', []))), 
                       health_data.get('avg_response_time', []), 
                       color=self.colors['info'], linewidth=2)
        axes[1, 1].axhline(y=5.0, color=self.colors['warning'], linestyle='--', alpha=0.7, label='Warning (5s)')
        axes[1, 1].set_title('Average Response Time', fontweight='bold')
        axes[1, 1].set_ylabel('Response Time (s)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            if timestamps:
                ax.tick_params(axis='x', rotation=45)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"System health plot saved to {save_path}")
        
        return fig
    
    def create_dashboard(self, performance_data: Dict[str, Any],
                        system_health: Dict[str, Any],
                        alerts: Dict[str, Any],
                        title: str = "Traffic Signal Optimization Dashboard",
                        save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Performance metrics (top row)
        metrics = ['wait_time', 'throughput', 'efficiency', 'confidence']
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[0, i])
            
            values = performance_data.get('trends', {}).get(metric, [])
            if values:
                ax.plot(values, color=self.colors['primary'], linewidth=2)
                ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add current value as text
                current_value = values[-1] if values else 0
                ax.text(0.02, 0.98, f'Current: {current_value:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Algorithm performance comparison (middle left)
        ax = fig.add_subplot(gs[1, :2])
        algorithm_data = performance_data.get('algorithm_performance', {})
        if algorithm_data:
            algorithms = list(algorithm_data.keys())
            scores = [data.get('avg_score', 0) for data in algorithm_data.values()]
            
            bars = ax.bar(algorithms, scores, color=[self.algorithm_colors.get(alg, self.colors['primary']) for alg in algorithms])
            ax.set_title('Algorithm Performance', fontweight='bold')
            ax.set_ylabel('Average Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        # System health (middle right)
        ax = fig.add_subplot(gs[1, 2:])
        health_metrics = ['cpu_usage', 'memory_usage', 'disk_usage']
        health_values = [system_health.get('current_health', {}).get(metric, 0) for metric in health_metrics]
        
        bars = ax.bar(health_metrics, health_values, 
                     color=[self.colors['success'] if v < 80 else self.colors['warning'] for v in health_values])
        ax.set_title('System Health', fontweight='bold')
        ax.set_ylabel('Usage (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax.axhline(y=80, color=self.colors['warning'], linestyle='--', alpha=0.7)
        ax.axhline(y=90, color=self.colors['warning'], linestyle='-', alpha=0.7)
        
        # Alerts summary (bottom left)
        ax = fig.add_subplot(gs[2, :2])
        alert_counts = alerts.get('by_severity', {})
        if alert_counts:
            severities = list(alert_counts.keys())
            counts = list(alert_counts.values())
            colors = [self.colors['warning'] if s == 'WARNING' else 
                     self.colors['warning'] if s == 'ERROR' else 
                     self.colors['info'] for s in severities]
            
            bars = ax.bar(severities, counts, color=colors)
            ax.set_title('Active Alerts by Severity', fontweight='bold')
            ax.set_ylabel('Number of Alerts')
            ax.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       str(count), ha='center', va='bottom')
        
        # Performance summary (bottom right)
        ax = fig.add_subplot(gs[2, 2:])
        ax.axis('off')
        
        # Create summary text
        total_optimizations = performance_data.get('total_optimizations', 0)
        avg_confidence = performance_data.get('avg_confidence', 0)
        avg_processing_time = performance_data.get('avg_processing_time', 0)
        active_alerts = alerts.get('total_active', 0)
        
        summary_text = f"""
        Performance Summary
        ───────────────────
        Total Optimizations: {total_optimizations:,}
        Average Confidence: {avg_confidence:.3f}
        Avg Processing Time: {avg_processing_time:.3f}s
        Active Alerts: {active_alerts}
        
        System Status: {'🟢 Healthy' if active_alerts == 0 else '🟡 Warning' if active_alerts < 5 else '🔴 Critical'}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def export_visualization_data(self, data: Dict[str, Any], 
                                 filepath: str, format: str = 'json'):
        """Export visualization data"""
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Convert to DataFrame and save
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Visualization data exported to {filepath}")
    
    def create_performance_report(self, performance_data: Dict[str, Any],
                                system_health: Dict[str, Any],
                                alerts: Dict[str, Any],
                                output_dir: str = None) -> str:
        """Create comprehensive performance report with visualizations"""
        
        if output_dir is None:
            output_dir = self.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create visualizations
        performance_trends_fig = self.plot_performance_trends(
            performance_data.get('trends', {}),
            title="Performance Trends Over Time"
        )
        
        algorithm_comparison_fig = self.plot_algorithm_comparison(
            performance_data.get('algorithm_performance', {}),
            title="Algorithm Performance Comparison"
        )
        
        system_health_fig = self.plot_system_health(
            system_health.get('trends_24h', {}),
            title="System Health Monitoring"
        )
        
        dashboard_fig = self.create_dashboard(
            performance_data, system_health, alerts,
            title="Traffic Signal Optimization Dashboard"
        )
        
        # Save visualizations
        performance_trends_fig.savefig(output_dir / f"performance_trends_{timestamp}.png", 
                                     dpi=300, bbox_inches='tight')
        algorithm_comparison_fig.savefig(output_dir / f"algorithm_comparison_{timestamp}.png", 
                                       dpi=300, bbox_inches='tight')
        system_health_fig.savefig(output_dir / f"system_health_{timestamp}.png", 
                                dpi=300, bbox_inches='tight')
        dashboard_fig.savefig(output_dir / f"dashboard_{timestamp}.png", 
                            dpi=300, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(performance_trends_fig)
        plt.close(algorithm_comparison_fig)
        plt.close(system_health_fig)
        plt.close(dashboard_fig)
        
        # Create HTML report
        html_report = self._create_html_report(
            performance_data, system_health, alerts, timestamp, output_dir
        )
        
        report_path = output_dir / f"performance_report_{timestamp}.html"
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        self.logger.info(f"Performance report created: {report_path}")
        
        return str(report_path)
    
    def _create_html_report(self, performance_data: Dict[str, Any],
                          system_health: Dict[str, Any],
                          alerts: Dict[str, Any],
                          timestamp: str, output_dir: Path) -> str:
        """Create HTML performance report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Traffic Signal Optimization Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2E86AB; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 3px; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .alert.warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .alert.error {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                .alert.info {{ background-color: #d1ecf1; border: 1px solid #bee5eb; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Traffic Signal Optimization Performance Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <div class="metric">
                    <strong>Total Optimizations:</strong> {performance_data.get('total_optimizations', 0):,}
                </div>
                <div class="metric">
                    <strong>Average Confidence:</strong> {performance_data.get('avg_confidence', 0):.3f}
                </div>
                <div class="metric">
                    <strong>Average Processing Time:</strong> {performance_data.get('avg_processing_time', 0):.3f}s
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Trends</h2>
                <img src="performance_trends_{timestamp}.png" alt="Performance Trends">
            </div>
            
            <div class="section">
                <h2>Algorithm Comparison</h2>
                <img src="algorithm_comparison_{timestamp}.png" alt="Algorithm Comparison">
            </div>
            
            <div class="section">
                <h2>System Health</h2>
                <img src="system_health_{timestamp}.png" alt="System Health">
            </div>
            
            <div class="section">
                <h2>Active Alerts</h2>
                <p><strong>Total Active Alerts:</strong> {alerts.get('total_active', 0)}</p>
        """
        
        # Add alerts
        for alert in alerts.get('recent_alerts', []):
            severity_class = alert['severity'].lower()
            html += f"""
                <div class="alert {severity_class}">
                    <strong>{alert['type']}:</strong> {alert['message']}<br>
                    <small>Time: {alert['timestamp']} | Intersection: {alert.get('intersection_id', 'N/A')}</small>
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Dashboard Overview</h2>
                <img src="dashboard_{timestamp}.png" alt="Dashboard Overview">
            </div>
        </body>
        </html>
        """.format(timestamp=timestamp)
        
        return html


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the visualizer
    visualizer = PerformanceVisualizer()
    
    # Generate sample data
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    
    performance_data = {
        'trends': {
            'wait_time': np.random.uniform(20, 40, 24),
            'throughput': np.random.uniform(400, 800, 24),
            'efficiency': np.random.uniform(0.6, 0.9, 24),
            'confidence': np.random.uniform(0.7, 0.95, 24)
        },
        'algorithm_performance': {
            'q_learning': {'wait_time': [25, 30, 28, 32], 'throughput': [600, 650, 620, 580]},
            'dynamic_programming': {'wait_time': [30, 35, 32, 38], 'throughput': [550, 600, 580, 520]},
            'websters_formula': {'wait_time': [35, 40, 38, 42], 'throughput': [500, 550, 520, 480]}
        },
        'total_optimizations': 1500,
        'avg_confidence': 0.85,
        'avg_processing_time': 0.5
    }
    
    system_health = {
        'current_health': {
            'cpu_usage': 45.2,
            'memory_usage': 62.1,
            'disk_usage': 38.5,
            'error_rate': 0.02,
            'avg_response_time': 0.8
        },
        'trends_24h': {
            'cpu_usage': np.random.uniform(30, 60, 24),
            'memory_usage': np.random.uniform(50, 70, 24),
            'disk_usage': np.random.uniform(30, 45, 24),
            'error_rate': np.random.uniform(0, 0.05, 24),
            'avg_response_time': np.random.uniform(0.5, 1.5, 24)
        }
    }
    
    alerts = {
        'total_active': 2,
        'by_severity': {'WARNING': 1, 'ERROR': 1},
        'recent_alerts': [
            {
                'alert_id': 'alert_001',
                'type': 'performance_degradation',
                'severity': 'WARNING',
                'message': 'High wait time detected',
                'timestamp': datetime.now().isoformat(),
                'intersection_id': 'junction_1'
            }
        ]
    }
    
    # Create visualizations
    print("Creating performance visualizations...")
    
    # Performance trends
    visualizer.plot_performance_trends(
        performance_data['trends'], timestamps,
        save_path="performance_trends.png"
    )
    
    # Algorithm comparison
    visualizer.plot_algorithm_comparison(
        performance_data['algorithm_performance'],
        save_path="algorithm_comparison.png"
    )
    
    # System health
    visualizer.plot_system_health(
        system_health['trends_24h'], timestamps,
        save_path="system_health.png"
    )
    
    # Dashboard
    visualizer.create_dashboard(
        performance_data, system_health, alerts,
        save_path="dashboard.png"
    )
    
    # Create comprehensive report
    report_path = visualizer.create_performance_report(
        performance_data, system_health, alerts
    )
    
    print(f"Visualizations created successfully!")
    print(f"Performance report: {report_path}")
