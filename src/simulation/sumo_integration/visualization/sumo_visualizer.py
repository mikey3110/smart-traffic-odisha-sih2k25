"""
SUMO Simulation Visualization Tools
Real-time visualization and analysis of simulation results
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo


class VisualizationType(Enum):
    """Visualization types"""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    COMPARISON = "comparison"
    HEATMAP = "heatmap"
    ANIMATION = "animation"


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    width: int = 1200
    height: int = 800
    dpi: int = 100
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    animation_interval: int = 100  # milliseconds
    save_format: str = "png"
    interactive: bool = True


class SumoVisualizer:
    """
    SUMO Simulation Visualization Tools
    
    Features:
    - Real-time dashboards
    - Historical trend analysis
    - Performance comparison charts
    - Traffic flow heatmaps
    - Animated visualizations
    - Export capabilities
    - Interactive plots
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        
        # Data storage
        self.simulation_data: List[Dict[str, Any]] = []
        self.performance_metrics: List[Dict[str, Any]] = []
        self.vehicle_data: List[Dict[str, Any]] = []
        self.intersection_data: List[Dict[str, Any]] = []
        
        # Real-time visualization
        self.is_visualizing = False
        self.visualization_thread = None
        self.figures: Dict[str, plt.Figure] = {}
        
        # Color mapping
        self.color_map = self._create_color_map()
        
        self.logger.info("SUMO Visualizer initialized")
    
    def _create_color_map(self) -> Dict[str, str]:
        """Create color mapping for different data types"""
        return {
            'vehicles': '#1f77b4',
            'waiting': '#ff7f0e',
            'moving': '#2ca02c',
            'intersections': '#d62728',
            'lanes': '#9467bd',
            'emissions': '#8c564b',
            'speed': '#e377c2',
            'waiting_time': '#7f7f7f'
        }
    
    def start_real_time_visualization(self):
        """Start real-time visualization"""
        if self.is_visualizing:
            return
        
        self.is_visualizing = True
        self.visualization_thread = threading.Thread(target=self._visualization_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        
        self.logger.info("Real-time visualization started")
    
    def stop_visualization(self):
        """Stop visualization"""
        self.is_visualizing = False
        if self.visualization_thread:
            self.visualization_thread.join(timeout=5)
        
        # Close all figures
        plt.close('all')
        
        self.logger.info("Visualization stopped")
    
    def _visualization_loop(self):
        """Main visualization loop"""
        while self.is_visualizing:
            try:
                # Update real-time plots
                self._update_real_time_plots()
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in visualization loop: {e}")
                time.sleep(5.0)
    
    def _update_real_time_plots(self):
        """Update real-time plots"""
        try:
            # This would be implemented to update real-time plots
            # For now, we'll just log the update
            self.logger.debug("Updating real-time plots")
            
        except Exception as e:
            self.logger.error(f"Error updating real-time plots: {e}")
    
    def create_dashboard(self, data: Dict[str, Any]) -> plt.Figure:
        """Create comprehensive simulation dashboard"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(self.config.width/100, self.config.height/100))
            fig.suptitle('SUMO Simulation Dashboard', fontsize=16, fontweight='bold')
            
            # Vehicle counts over time
            self._plot_vehicle_counts(axes[0, 0], data)
            
            # Waiting time distribution
            self._plot_waiting_time_distribution(axes[0, 1], data)
            
            # Speed distribution
            self._plot_speed_distribution(axes[0, 2], data)
            
            # Intersection performance
            self._plot_intersection_performance(axes[1, 0], data)
            
            # Emissions over time
            self._plot_emissions_over_time(axes[1, 1], data)
            
            # Traffic flow heatmap
            self._plot_traffic_flow_heatmap(axes[1, 2], data)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
            return None
    
    def _plot_vehicle_counts(self, ax, data: Dict[str, Any]):
        """Plot vehicle counts over time"""
        try:
            if 'performance_metrics' in data:
                metrics = data['performance_metrics']
                times = list(range(len(metrics)))
                total_vehicles = [m.get('total_vehicles', 0) for m in metrics]
                waiting_vehicles = [m.get('waiting_vehicles', 0) for m in metrics]
                
                ax.plot(times, total_vehicles, label='Total Vehicles', color=self.color_map['vehicles'])
                ax.plot(times, waiting_vehicles, label='Waiting Vehicles', color=self.color_map['waiting'])
                ax.set_xlabel('Time (steps)')
                ax.set_ylabel('Vehicle Count')
                ax.set_title('Vehicle Counts Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Error plotting vehicle counts: {e}")
    
    def _plot_waiting_time_distribution(self, ax, data: Dict[str, Any]):
        """Plot waiting time distribution"""
        try:
            if 'vehicles' in data:
                vehicles = data['vehicles']
                waiting_times = [v.get('waiting_time', 0) for v in vehicles]
                
                ax.hist(waiting_times, bins=20, alpha=0.7, color=self.color_map['waiting_time'])
                ax.set_xlabel('Waiting Time (seconds)')
                ax.set_ylabel('Frequency')
                ax.set_title('Waiting Time Distribution')
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Error plotting waiting time distribution: {e}")
    
    def _plot_speed_distribution(self, ax, data: Dict[str, Any]):
        """Plot speed distribution"""
        try:
            if 'vehicles' in data:
                vehicles = data['vehicles']
                speeds = [v.get('speed', 0) for v in vehicles]
                
                ax.hist(speeds, bins=20, alpha=0.7, color=self.color_map['speed'])
                ax.set_xlabel('Speed (m/s)')
                ax.set_ylabel('Frequency')
                ax.set_title('Speed Distribution')
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Error plotting speed distribution: {e}")
    
    def _plot_intersection_performance(self, ax, data: Dict[str, Any]):
        """Plot intersection performance"""
        try:
            if 'intersections' in data:
                intersections = data['intersections']
                intersection_ids = [i.get('id', '') for i in intersections]
                waiting_counts = [i.get('waiting_vehicles', 0) for i in intersections]
                
                ax.bar(intersection_ids, waiting_counts, color=self.color_map['intersections'])
                ax.set_xlabel('Intersection ID')
                ax.set_ylabel('Waiting Vehicles')
                ax.set_title('Intersection Performance')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Error plotting intersection performance: {e}")
    
    def _plot_emissions_over_time(self, ax, data: Dict[str, Any]):
        """Plot emissions over time"""
        try:
            if 'emissions' in data:
                emissions = data['emissions']
                times = list(range(len(emissions)))
                co2_emissions = [e.get('total_co2', 0) for e in emissions]
                
                ax.plot(times, co2_emissions, color=self.color_map['emissions'])
                ax.set_xlabel('Time (steps)')
                ax.set_ylabel('CO2 Emissions')
                ax.set_title('CO2 Emissions Over Time')
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Error plotting emissions over time: {e}")
    
    def _plot_traffic_flow_heatmap(self, ax, data: Dict[str, Any]):
        """Plot traffic flow heatmap"""
        try:
            if 'lanes' in data:
                lanes = data['lanes']
                lane_ids = [l.get('id', '') for l in lanes]
                vehicle_counts = [l.get('vehicle_count', 0) for l in lanes]
                
                # Create heatmap data
                heatmap_data = np.array(vehicle_counts).reshape(1, -1)
                
                im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(lane_ids)))
                ax.set_xticklabels(lane_ids, rotation=45)
                ax.set_yticks([0])
                ax.set_yticklabels(['Vehicle Count'])
                ax.set_title('Traffic Flow Heatmap')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
            
        except Exception as e:
            self.logger.error(f"Error plotting traffic flow heatmap: {e}")
    
    def create_performance_comparison(self, data_sets: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """Create performance comparison chart"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Performance Comparison', fontsize=16, fontweight='bold')
            
            # Compare different scenarios
            scenarios = list(data_sets.keys())
            
            # Total vehicles comparison
            self._plot_scenario_comparison(axes[0, 0], data_sets, 'total_vehicles', 'Total Vehicles')
            
            # Waiting time comparison
            self._plot_scenario_comparison(axes[0, 1], data_sets, 'average_waiting_time', 'Average Waiting Time')
            
            # Speed comparison
            self._plot_scenario_comparison(axes[1, 0], data_sets, 'average_speed', 'Average Speed')
            
            # Emissions comparison
            self._plot_scenario_comparison(axes[1, 1], data_sets, 'total_co2', 'Total CO2 Emissions')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance comparison: {e}")
            return None
    
    def _plot_scenario_comparison(self, ax, data_sets: Dict[str, Dict[str, Any]], metric: str, title: str):
        """Plot scenario comparison for specific metric"""
        try:
            scenarios = list(data_sets.keys())
            values = []
            
            for scenario in scenarios:
                data = data_sets[scenario]
                if 'performance_metrics' in data:
                    metrics = data['performance_metrics']
                    if metrics and len(metrics) > 0:
                        # Get average value
                        avg_value = np.mean([m.get(metric, 0) for m in metrics])
                        values.append(avg_value)
                    else:
                        values.append(0)
                else:
                    values.append(0)
            
            bars = ax.bar(scenarios, values, color=[self.color_map['vehicles'], self.color_map['waiting'], 
                                                   self.color_map['moving'], self.color_map['intersections']])
            ax.set_title(title)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
            
        except Exception as e:
            self.logger.error(f"Error plotting scenario comparison: {e}")
    
    def create_interactive_dashboard(self, data: Dict[str, Any]) -> str:
        """Create interactive Plotly dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Vehicle Counts', 'Waiting Time Distribution', 
                              'Speed Distribution', 'Intersection Performance',
                              'Emissions Over Time', 'Traffic Flow Heatmap'),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "heatmap"}]]
            )
            
            # Add vehicle counts plot
            if 'performance_metrics' in data:
                metrics = data['performance_metrics']
                times = list(range(len(metrics)))
                total_vehicles = [m.get('total_vehicles', 0) for m in metrics]
                waiting_vehicles = [m.get('waiting_vehicles', 0) for m in metrics]
                
                fig.add_trace(
                    go.Scatter(x=times, y=total_vehicles, name='Total Vehicles', line=dict(color='blue')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=times, y=waiting_vehicles, name='Waiting Vehicles', line=dict(color='red')),
                    row=1, col=1
                )
            
            # Add waiting time histogram
            if 'vehicles' in data:
                vehicles = data['vehicles']
                waiting_times = [v.get('waiting_time', 0) for v in vehicles]
                
                fig.add_trace(
                    go.Histogram(x=waiting_times, name='Waiting Time', nbinsx=20),
                    row=1, col=2
                )
            
            # Add speed histogram
            if 'vehicles' in data:
                vehicles = data['vehicles']
                speeds = [v.get('speed', 0) for v in vehicles]
                
                fig.add_trace(
                    go.Histogram(x=speeds, name='Speed', nbinsx=20),
                    row=2, col=1
                )
            
            # Add intersection performance bar chart
            if 'intersections' in data:
                intersections = data['intersections']
                intersection_ids = [i.get('id', '') for i in intersections]
                waiting_counts = [i.get('waiting_vehicles', 0) for i in intersections]
                
                fig.add_trace(
                    go.Bar(x=intersection_ids, y=waiting_counts, name='Waiting Vehicles'),
                    row=2, col=2
                )
            
            # Add emissions plot
            if 'emissions' in data:
                emissions = data['emissions']
                times = list(range(len(emissions)))
                co2_emissions = [e.get('total_co2', 0) for e in emissions]
                
                fig.add_trace(
                    go.Scatter(x=times, y=co2_emissions, name='CO2 Emissions', line=dict(color='green')),
                    row=3, col=1
                )
            
            # Add traffic flow heatmap
            if 'lanes' in data:
                lanes = data['lanes']
                lane_ids = [l.get('id', '') for l in lanes]
                vehicle_counts = [l.get('vehicle_count', 0) for l in lanes]
                
                fig.add_trace(
                    go.Heatmap(z=[vehicle_counts], x=lane_ids, name='Traffic Flow'),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="SUMO Simulation Interactive Dashboard",
                showlegend=True,
                height=800,
                width=1200
            )
            
            # Save as HTML
            html_file = "sumo_dashboard.html"
            pyo.plot(fig, filename=html_file, auto_open=False)
            
            return html_file
            
        except Exception as e:
            self.logger.error(f"Error creating interactive dashboard: {e}")
            return None
    
    def create_animation(self, data_series: List[Dict[str, Any]], output_file: str = "simulation_animation.gif"):
        """Create animated visualization"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            def animate(frame):
                ax.clear()
                
                if frame < len(data_series):
                    data = data_series[frame]
                    
                    # Plot current state
                    if 'vehicles' in data:
                        vehicles = data['vehicles']
                        x_positions = [v.get('position', [0, 0])[0] for v in vehicles]
                        y_positions = [v.get('position', [0, 0])[1] for v in vehicles]
                        speeds = [v.get('speed', 0) for v in vehicles]
                        
                        scatter = ax.scatter(x_positions, y_positions, c=speeds, 
                                           cmap='viridis', alpha=0.7)
                        ax.set_title(f'Simulation Step {frame}')
                        ax.set_xlabel('X Position')
                        ax.set_ylabel('Y Position')
                        ax.grid(True, alpha=0.3)
            
            # Create animation
            anim = animation.FuncAnimation(fig, animate, frames=len(data_series), 
                                        interval=self.config.animation_interval, repeat=True)
            
            # Save animation
            anim.save(output_file, writer='pillow', fps=10)
            
            self.logger.info(f"Animation saved to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error creating animation: {e}")
            return None
    
    def export_visualization(self, fig: plt.Figure, filepath: str, format: str = None):
        """Export visualization to file"""
        try:
            if format is None:
                format = self.config.save_format
            
            fig.savefig(filepath, format=format, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"Visualization exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting visualization: {e}")
    
    def add_data_point(self, data: Dict[str, Any]):
        """Add data point for visualization"""
        try:
            self.simulation_data.append(data)
            
            # Keep only recent data
            if len(self.simulation_data) > 1000:
                self.simulation_data = self.simulation_data[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error adding data point: {e}")
    
    def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get visualization statistics"""
        return {
            'is_visualizing': self.is_visualizing,
            'data_points': len(self.simulation_data),
            'performance_metrics': len(self.performance_metrics),
            'vehicle_data_points': len(self.vehicle_data),
            'intersection_data_points': len(self.intersection_data),
            'active_figures': len(self.figures)
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create visualizer
    config = VisualizationConfig(interactive=True)
    visualizer = SumoVisualizer(config)
    
    # Sample data
    sample_data = {
        'performance_metrics': [
            {'total_vehicles': 50, 'waiting_vehicles': 10, 'average_speed': 30.0},
            {'total_vehicles': 60, 'waiting_vehicles': 15, 'average_speed': 25.0},
            {'total_vehicles': 45, 'waiting_vehicles': 8, 'average_speed': 35.0}
        ],
        'vehicles': [
            {'position': [100, 200], 'speed': 30.0, 'waiting_time': 0.0},
            {'position': [150, 250], 'speed': 25.0, 'waiting_time': 5.0},
            {'position': [200, 300], 'speed': 35.0, 'waiting_time': 0.0}
        ],
        'intersections': [
            {'id': 'intersection_1', 'waiting_vehicles': 5},
            {'id': 'intersection_2', 'waiting_vehicles': 3}
        ],
        'lanes': [
            {'id': 'lane_1', 'vehicle_count': 10},
            {'id': 'lane_2', 'vehicle_count': 8}
        ],
        'emissions': [
            {'total_co2': 100.0},
            {'total_co2': 120.0},
            {'total_co2': 90.0}
        ]
    }
    
    # Create dashboard
    dashboard = visualizer.create_dashboard(sample_data)
    if dashboard:
        visualizer.export_visualization(dashboard, "sumo_dashboard.png")
    
    # Create interactive dashboard
    html_file = visualizer.create_interactive_dashboard(sample_data)
    if html_file:
        print(f"Interactive dashboard created: {html_file}")
    
    # Get statistics
    stats = visualizer.get_visualization_statistics()
    print(f"Visualization statistics: {stats}")
