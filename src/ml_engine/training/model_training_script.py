"""
Model Training and Evaluation Scripts for ML Traffic Signal Optimization
Comprehensive training pipeline for all ML models
"""

import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.ml_config import MLConfig, get_config, load_config
from prediction.enhanced_traffic_predictor import EnhancedTrafficPredictor
from algorithms.enhanced_q_learning_optimizer import EnhancedQLearningOptimizer
from algorithms.enhanced_dynamic_programming_optimizer import EnhancedDynamicProgrammingOptimizer
from algorithms.enhanced_websters_formula_optimizer import EnhancedWebstersFormulaOptimizer
from metrics.enhanced_performance_metrics import EnhancedPerformanceMetrics
from monitoring.enhanced_monitoring import EnhancedMonitoring


class ModelTrainer:
    """
    Comprehensive model training and evaluation system
    
    Features:
    - Data generation and preprocessing
    - Model training for all algorithms
    - Cross-validation and evaluation
    - Hyperparameter tuning
    - Model comparison and selection
    - Performance visualization
    - Model persistence and versioning
    """
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.traffic_predictor = EnhancedTrafficPredictor(self.config.traffic_prediction)
        self.performance_metrics = EnhancedPerformanceMetrics(self.config.performance_metrics)
        self.monitoring = EnhancedMonitoring(self.config.logging)
        
        # Training results
        self.training_results: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, Any]] = {}
        
        # Create directories
        self._create_directories()
        
        self.logger.info("Model trainer initialized")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.model_save_path,
            self.config.data_save_path,
            "logs",
            "results",
            "visualizations"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def generate_training_data(self, num_samples: int = 10000, 
                             duration_days: int = 30) -> pd.DataFrame:
        """Generate synthetic training data"""
        self.logger.info(f"Generating {num_samples} training samples...")
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=duration_days)
        timestamps = pd.date_range(
            start=start_date, 
            end=datetime.now(), 
            periods=num_samples
        )
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Generate realistic traffic patterns
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Base traffic patterns
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                base_traffic = np.random.poisson(15)
            elif 10 <= hour <= 16:  # Daytime
                base_traffic = np.random.poisson(10)
            else:  # Night time
                base_traffic = np.random.poisson(3)
            
            # Weekend adjustment
            if day_of_week >= 5:  # Weekend
                base_traffic = int(base_traffic * 0.7)
            
            # Generate lane counts
            lane_counts = {
                'north_lane': max(0, base_traffic + np.random.randint(-5, 6)),
                'south_lane': max(0, base_traffic + np.random.randint(-5, 6)),
                'east_lane': max(0, base_traffic + np.random.randint(-5, 6)),
                'west_lane': max(0, base_traffic + np.random.randint(-5, 6))
            }
            
            # Generate other features
            avg_speed = max(10, min(60, 50 - sum(lane_counts.values()) * 0.5 + np.random.normal(0, 5)))
            
            weather_conditions = ['clear', 'cloudy', 'rainy', 'foggy', 'stormy', 'snowy']
            weather_condition = np.random.choice(weather_conditions, p=[0.4, 0.3, 0.15, 0.05, 0.05, 0.05])
            
            # Environmental factors
            temperature = 20 + np.random.normal(0, 8)
            humidity = 50 + np.random.normal(0, 15)
            visibility = 10 + np.random.normal(0, 2)
            
            # Generate signal timings (ground truth)
            total_vehicles = sum(lane_counts.values())
            if total_vehicles > 20:
                cycle_time = 90
            elif total_vehicles > 10:
                cycle_time = 60
            else:
                cycle_time = 45
            
            # Allocate green times based on traffic
            north_south_ratio = (lane_counts['north_lane'] + lane_counts['south_lane']) / max(total_vehicles, 1)
            east_west_ratio = (lane_counts['east_lane'] + lane_counts['west_lane']) / max(total_vehicles, 1)
            
            green_time = cycle_time - 10  # Reserve 10 seconds for yellow/red
            north_south_green = int(green_time * north_south_ratio)
            east_west_green = int(green_time * east_west_ratio)
            
            signal_timings = {
                'north_lane': max(15, north_south_green),
                'south_lane': max(15, north_south_green),
                'east_lane': max(15, east_west_green),
                'west_lane': max(15, east_west_green)
            }
            
            # Calculate performance metrics
            wait_time = max(0, total_vehicles * 2 + np.random.normal(0, 5))
            throughput = max(0, total_vehicles * 0.8 + np.random.normal(0, 10))
            efficiency = max(0, min(1, 0.5 + (throughput / 1000) - (wait_time / 100)))
            
            data.append({
                'timestamp': timestamp,
                'intersection_id': f'junction_{i % 10}',  # 10 different intersections
                'north_lane': lane_counts['north_lane'],
                'south_lane': lane_counts['south_lane'],
                'east_lane': lane_counts['east_lane'],
                'west_lane': lane_counts['west_lane'],
                'avg_speed': avg_speed,
                'weather_condition': weather_condition,
                'temperature': temperature,
                'humidity': humidity,
                'visibility': visibility,
                'north_lane_timing': signal_timings['north_lane'],
                'south_lane_timing': signal_timings['south_lane'],
                'east_lane_timing': signal_timings['east_lane'],
                'west_lane_timing': signal_timings['west_lane'],
                'wait_time': wait_time,
                'throughput': throughput,
                'efficiency': efficiency
            })
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated training data: {df.shape}")
        
        # Save training data
        data_path = os.path.join(self.config.data_save_path, "training_data.csv")
        df.to_csv(data_path, index=False)
        self.logger.info(f"Training data saved to {data_path}")
        
        return df
    
    def train_traffic_predictor(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train traffic prediction models"""
        self.logger.info("Training traffic prediction models...")
        
        start_time = time.time()
        
        # Train models
        training_results = self.traffic_predictor.train_models(training_data)
        
        # Evaluate models
        performance = self.traffic_predictor.evaluate_models(training_data)
        
        training_time = time.time() - start_time
        
        # Save models
        model_path = os.path.join(self.config.model_save_path, "traffic_predictor")
        self.traffic_predictor.save_models(model_path)
        
        result = {
            'training_results': training_results,
            'performance': performance,
            'training_time': training_time,
            'model_path': model_path
        }
        
        self.training_results['traffic_predictor'] = result
        self.logger.info(f"Traffic predictor training completed in {training_time:.2f}s")
        
        return result
    
    def train_q_learning_optimizer(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train Q-Learning optimizer"""
        self.logger.info("Training Q-Learning optimizer...")
        
        start_time = time.time()
        
        # Initialize optimizer
        q_learning = EnhancedQLearningOptimizer(self.config.q_learning)
        
        # Prepare training episodes
        episodes = self._prepare_q_learning_episodes(training_data)
        
        # Train model
        training_results = []
        for episode in episodes:
            result = q_learning.train_episode(episode)
            training_results.append(result)
        
        training_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(self.config.model_save_path, "q_learning_model.pkl")
        q_learning.save_model(model_path)
        
        result = {
            'training_results': training_results,
            'training_time': training_time,
            'model_path': model_path,
            'final_epsilon': q_learning.epsilon,
            'total_episodes': len(episodes)
        }
        
        self.training_results['q_learning'] = result
        self.logger.info(f"Q-Learning training completed in {training_time:.2f}s")
        
        return result
    
    def _prepare_q_learning_episodes(self, training_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare training episodes for Q-Learning"""
        episodes = []
        
        # Group by intersection and time windows
        for intersection_id in training_data['intersection_id'].unique():
            intersection_data = training_data[training_data['intersection_id'] == intersection_id].sort_values('timestamp')
            
            # Create episodes of 10 consecutive time steps
            for i in range(0, len(intersection_data) - 10, 5):
                episode_data = intersection_data.iloc[i:i+10]
                
                episode = {
                    'intersection_id': intersection_id,
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'next_states': []
                }
                
                for _, row in episode_data.iterrows():
                    state = {
                        'lane_counts': {
                            'north_lane': row['north_lane'],
                            'south_lane': row['south_lane'],
                            'east_lane': row['east_lane'],
                            'west_lane': row['west_lane']
                        },
                        'avg_speed': row['avg_speed'],
                        'weather_condition': row['weather_condition'],
                        'temperature': row['temperature'],
                        'humidity': row['humidity'],
                        'visibility': row['visibility'],
                        'timestamp': row['timestamp']
                    }
                    
                    action = {
                        'north_lane_timing': row['north_lane_timing'],
                        'south_lane_timing': row['south_lane_timing'],
                        'east_lane_timing': row['east_lane_timing'],
                        'west_lane_timing': row['west_lane_timing']
                    }
                    
                    reward = self._calculate_reward(row)
                    
                    episode['states'].append(state)
                    episode['actions'].append(action)
                    episode['rewards'].append(reward)
                
                # Calculate next states
                for i in range(len(episode['states']) - 1):
                    episode['next_states'].append(episode['states'][i + 1])
                episode['next_states'].append(episode['states'][-1])  # Last state repeats
                
                episodes.append(episode)
        
        return episodes
    
    def _calculate_reward(self, row: pd.Series) -> float:
        """Calculate reward for Q-Learning"""
        # Reward based on efficiency and penalty for wait time
        efficiency_reward = row['efficiency'] * 10
        wait_time_penalty = -row['wait_time'] * 0.1
        throughput_reward = row['throughput'] * 0.01
        
        return efficiency_reward + wait_time_penalty + throughput_reward
    
    def train_all_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train all models"""
        self.logger.info("Starting comprehensive model training...")
        
        start_time = time.time()
        
        # Train traffic predictor
        predictor_result = self.train_traffic_predictor(training_data)
        
        # Train Q-Learning optimizer
        q_learning_result = self.train_q_learning_optimizer(training_data)
        
        # Note: Dynamic Programming and Webster's Formula don't require training
        # as they are rule-based algorithms
        
        total_training_time = time.time() - start_time
        
        # Compile results
        all_results = {
            'traffic_predictor': predictor_result,
            'q_learning': q_learning_result,
            'total_training_time': total_training_time,
            'training_data_shape': training_data.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save training results
        results_path = os.path.join("results", f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self.logger.info(f"All models trained in {total_training_time:.2f}s")
        self.logger.info(f"Results saved to {results_path}")
        
        return all_results
    
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate all trained models"""
        self.logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        # Evaluate traffic predictor
        if 'traffic_predictor' in self.training_results:
            predictor_performance = self.traffic_predictor.get_model_performance()
            evaluation_results['traffic_predictor'] = predictor_performance
        
        # Evaluate Q-Learning optimizer
        if 'q_learning' in self.training_results:
            q_learning_stats = self.training_results['q_learning']
            evaluation_results['q_learning'] = {
                'training_episodes': q_learning_stats['total_episodes'],
                'final_epsilon': q_learning_stats['final_epsilon'],
                'training_time': q_learning_stats['training_time']
            }
        
        # Evaluate rule-based algorithms
        evaluation_results['dynamic_programming'] = {
            'type': 'rule_based',
            'status': 'ready',
            'description': 'Dynamic programming optimizer ready for use'
        }
        
        evaluation_results['websters_formula'] = {
            'type': 'rule_based',
            'status': 'ready',
            'description': 'Webster\'s formula optimizer ready for use'
        }
        
        # Save evaluation results
        eval_path = os.path.join("results", f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        self.logger.info(f"Model evaluation completed. Results saved to {eval_path}")
        
        return evaluation_results
    
    def create_training_report(self, training_results: Dict[str, Any], 
                             evaluation_results: Dict[str, Any]) -> str:
        """Create comprehensive training report"""
        report = f"""
# ML Traffic Signal Optimization - Training Report

## Training Summary
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Training Time**: {training_results.get('total_training_time', 0):.2f} seconds
- **Training Data Size**: {training_results.get('training_data_shape', (0, 0))[0]} samples

## Model Performance

### Traffic Predictor
"""
        
        if 'traffic_predictor' in training_results:
            predictor_result = training_results['traffic_predictor']
            report += f"""
- **Training Time**: {predictor_result.get('training_time', 0):.2f} seconds
- **Model Path**: {predictor_result.get('model_path', 'N/A')}
- **Performance**: Available in evaluation results
"""
        
        report += """
### Q-Learning Optimizer
"""
        
        if 'q_learning' in training_results:
            ql_result = training_results['q_learning']
            report += f"""
- **Training Time**: {ql_result.get('training_time', 0):.2f} seconds
- **Total Episodes**: {ql_result.get('total_episodes', 0)}
- **Final Epsilon**: {ql_result.get('final_epsilon', 0):.4f}
- **Model Path**: {ql_result.get('model_path', 'N/A')}
"""
        
        report += """
### Rule-Based Algorithms
- **Dynamic Programming**: Ready for use
- **Webster's Formula**: Ready for use

## Recommendations
1. Deploy traffic predictor for real-time flow prediction
2. Use Q-Learning for adaptive optimization in high-traffic scenarios
3. Use Webster's Formula as fallback for stable performance
4. Monitor model performance in production environment

## Next Steps
1. Deploy models to production environment
2. Set up continuous monitoring
3. Implement A/B testing for algorithm comparison
4. Schedule periodic model retraining
"""
        
        # Save report
        report_path = os.path.join("results", f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Training report saved to {report_path}")
        
        return report


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train ML models for traffic signal optimization')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--days', type=int, default=30, help='Duration of training data in days')
    parser.add_argument('--skip-data-generation', action='store_true', help='Skip data generation')
    parser.add_argument('--data-file', type=str, help='Path to existing training data file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    try:
        # Generate or load training data
        if args.data_file and os.path.exists(args.data_file):
            logging.info(f"Loading training data from {args.data_file}")
            training_data = pd.read_csv(args.data_file)
        elif not args.skip_data_generation:
            training_data = trainer.generate_training_data(args.samples, args.days)
        else:
            raise ValueError("No training data available and data generation skipped")
        
        # Train all models
        training_results = trainer.train_all_models(training_data)
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(training_data)
        
        # Create training report
        report = trainer.create_training_report(training_results, evaluation_results)
        
        print("Training completed successfully!")
        print(f"Training results: {training_results}")
        print(f"Evaluation results: {evaluation_results}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
