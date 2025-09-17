#!/usr/bin/env python3
"""
Computer Vision Accuracy Validation Script
Validates YOLO model accuracy on test images and generates performance report
"""

import os
import cv2
import time
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from datetime import datetime

class CVValidator:
    def __init__(self, model_path, test_images_dir):
        self.model_path = model_path
        self.test_images_dir = Path(test_images_dir)
        self.model = None
        self.results = []
        self.performance_metrics = {}
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def validate_accuracy(self):
        """Run accuracy validation on test images"""
        if not self.model:
            print("❌ Model not loaded. Please load model first.")
            return False
        
        print(f"\n🔍 Running accuracy validation on {len(list(self.test_images_dir.glob('*.jpg')))} test images...")
        
        total_vehicles_detected = 0
        total_images = 0
        processing_times = []
        
        for img_path in self.test_images_dir.glob('*.jpg'):
            print(f"Processing: {img_path.name}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️  Could not load {img_path.name}")
                continue
            
            # Run detection
            start_time = time.time()
            results = self.model(image, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Count vehicles
            vehicle_count = len(results[0].boxes) if results[0].boxes is not None else 0
            total_vehicles_detected += vehicle_count
            total_images += 1
            
            # Store results
            result_data = {
                'image_name': img_path.name,
                'vehicles_detected': vehicle_count,
                'processing_time': processing_time,
                'image_size': image.shape[:2],
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result_data)
            
            print(f"  Vehicles detected: {vehicle_count}, Time: {processing_time:.3f}s")
        
        # Calculate metrics
        self.performance_metrics = {
            'total_images_processed': total_images,
            'total_vehicles_detected': total_vehicles_detected,
            'average_vehicles_per_image': total_vehicles_detected / total_images if total_images > 0 else 0,
            'average_processing_time': np.mean(processing_times) if processing_times else 0,
            'fps': 1.0 / np.mean(processing_times) if processing_times else 0,
            'min_processing_time': np.min(processing_times) if processing_times else 0,
            'max_processing_time': np.max(processing_times) if processing_times else 0,
            'model_path': self.model_path,
            'test_images_dir': str(self.test_images_dir),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        print(f"\n📊 Validation Results:")
        print(f"  Total images processed: {total_images}")
        print(f"  Total vehicles detected: {total_vehicles_detected}")
        print(f"  Average vehicles per image: {self.performance_metrics['average_vehicles_per_image']:.2f}")
        print(f"  Average processing time: {self.performance_metrics['average_processing_time']:.3f}s")
        print(f"  FPS: {self.performance_metrics['fps']:.2f}")
        
        return True
    
    def generate_report(self, output_file="docs/cv_report.md"):
        """Generate comprehensive CV validation report"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Computer Vision Accuracy Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Model Information\n")
            f.write(f"- **Model Path:** {self.performance_metrics['model_path']}\n")
            f.write(f"- **Test Images:** {self.performance_metrics['test_images_dir']}\n")
            f.write(f"- **Total Images:** {self.performance_metrics['total_images_processed']}\n\n")
            
            f.write("## Performance Metrics\n")
            f.write(f"- **Total Vehicles Detected:** {self.performance_metrics['total_vehicles_detected']}\n")
            f.write(f"- **Average Vehicles per Image:** {self.performance_metrics['average_vehicles_per_image']:.2f}\n")
            f.write(f"- **Average Processing Time:** {self.performance_metrics['average_processing_time']:.3f}s\n")
            f.write(f"- **FPS (Frames Per Second):** {self.performance_metrics['fps']:.2f}\n")
            f.write(f"- **Min Processing Time:** {self.performance_metrics['min_processing_time']:.3f}s\n")
            f.write(f"- **Max Processing Time:** {self.performance_metrics['max_processing_time']:.3f}s\n\n")
            
            f.write("## Performance Analysis\n")
            if self.performance_metrics['fps'] >= 15:
                f.write("✅ **FPS Performance:** Excellent (≥15 FPS)\n")
            elif self.performance_metrics['fps'] >= 10:
                f.write("⚠️  **FPS Performance:** Good (10-15 FPS)\n")
            else:
                f.write("❌ **FPS Performance:** Needs improvement (<10 FPS)\n")
            
            f.write(f"- **Processing Speed:** {self.performance_metrics['fps']:.2f} FPS\n")
            f.write(f"- **Detection Rate:** {self.performance_metrics['average_vehicles_per_image']:.2f} vehicles per image\n\n")
            
            f.write("## Detailed Results\n")
            f.write("| Image | Vehicles | Processing Time (s) | Image Size |\n")
            f.write("|-------|----------|-------------------|------------|\n")
            
            for result in self.results:
                f.write(f"| {result['image_name']} | {result['vehicles_detected']} | {result['processing_time']:.3f} | {result['image_size'][0]}x{result['image_size'][1]} |\n")
            
            f.write("\n## Recommendations\n")
            if self.performance_metrics['fps'] < 15:
                f.write("- Consider model optimization (ONNX/TensorRT conversion)\n")
                f.write("- Reduce input image resolution if needed\n")
                f.write("- Use GPU acceleration if available\n")
            
            f.write("- Monitor memory usage during extended operation\n")
            f.write("- Test with different lighting conditions\n")
            f.write("- Validate detection accuracy with manual counting\n\n")
            
            f.write("## Hardware Requirements\n")
            f.write(f"- **Minimum FPS:** 15 FPS for real-time processing\n")
            f.write(f"- **Current Performance:** {self.performance_metrics['fps']:.2f} FPS\n")
            f.write(f"- **Recommended:** GPU with CUDA support for optimal performance\n\n")
        
        print(f"📄 Report generated: {output_file}")
        
        # Also save JSON results
        json_file = output_file.replace('.md', '_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'performance_metrics': self.performance_metrics,
                'detailed_results': self.results
            }, f, indent=2)
        
        print(f"📄 JSON results saved: {json_file}")
    
    def run_validation(self):
        """Run complete validation pipeline"""
        print("🚀 Starting Computer Vision Validation")
        print("=" * 50)
        
        if not self.load_model():
            return False
        
        if not self.validate_accuracy():
            return False
        
        self.generate_report()
        
        print("\n✅ Validation completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Validate Computer Vision Model Accuracy')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model file')
    parser.add_argument('--images', default='src/computer_vision/test_images', help='Path to test images directory')
    parser.add_argument('--output', default='docs/cv_report.md', help='Output report file')
    
    args = parser.parse_args()
    
    validator = CVValidator(args.model, args.images)
    success = validator.run_validation()
    
    if success:
        print(f"\n🎯 Validation Summary:")
        print(f"  Model: {args.model}")
        print(f"  Images: {args.images}")
        print(f"  FPS: {validator.performance_metrics['fps']:.2f}")
        print(f"  Report: {args.output}")
    else:
        print("\n❌ Validation failed!")
        exit(1)

if __name__ == "__main__":
    main()
