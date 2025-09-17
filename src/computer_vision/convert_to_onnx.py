#!/usr/bin/env python3
"""
YOLO Model Optimization Script
Converts YOLO model to ONNX format for improved performance
"""

import os
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def convert_yolo_to_onnx(model_path, output_dir="src/computer_vision/models"):
    """
    Convert YOLO model to ONNX format for optimization
    """
    print("🚀 Starting YOLO to ONNX conversion")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the model
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        
        # Get model info
        print(f"Model type: {type(model.model)}")
        print(f"Model device: {next(model.model.parameters()).device}")
        
        # Convert to ONNX
        print("\n🔄 Converting to ONNX format...")
        start_time = time.time()
        
        # Export to ONNX with optimization
        onnx_path = model.export(
            format='onnx',
            optimize=True,
            imgsz=640,
            device='cpu',  # Use CPU for compatibility
            verbose=True
        )
        
        conversion_time = time.time() - start_time
        print(f"✅ ONNX conversion completed in {conversion_time:.2f}s")
        print(f"ONNX model saved to: {onnx_path}")
        
        # Move to our models directory
        target_path = os.path.join(output_dir, "yolov8n-optimized.onnx")
        if os.path.exists(onnx_path):
            os.rename(onnx_path, target_path)
            print(f"✅ Model moved to: {target_path}")
        
        # Test the ONNX model
        print("\n🧪 Testing ONNX model...")
        test_onnx_model(target_path)
        
        return target_path
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None

def test_onnx_model(onnx_path):
    """Test the converted ONNX model"""
    try:
        # Load ONNX model
        onnx_model = YOLO(onnx_path)
        
        # Create a test image (random)
        import numpy as np
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test inference
        start_time = time.time()
        results = onnx_model(test_image)
        inference_time = time.time() - start_time
        
        print(f"✅ ONNX model test successful")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  FPS: {1.0/inference_time:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX model test failed: {e}")
        return False

def benchmark_models(pytorch_path, onnx_path):
    """Benchmark PyTorch vs ONNX model performance"""
    print("\n📊 Benchmarking model performance...")
    print("=" * 50)
    
    import numpy as np
    import time
    
    # Create test images
    test_images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(10)]
    
    # Test PyTorch model
    print("Testing PyTorch model...")
    pytorch_model = YOLO(pytorch_path)
    pytorch_times = []
    
    for img in test_images:
        start_time = time.time()
        results = pytorch_model(img)
        pytorch_times.append(time.time() - start_time)
    
    # Test ONNX model
    print("Testing ONNX model...")
    onnx_model = YOLO(onnx_path)
    onnx_times = []
    
    for img in test_images:
        start_time = time.time()
        results = onnx_model(img)
        onnx_times.append(time.time() - start_time)
    
    # Calculate metrics
    pytorch_avg = np.mean(pytorch_times)
    onnx_avg = np.mean(onnx_times)
    speedup = pytorch_avg / onnx_avg
    
    print(f"\n📈 Benchmark Results:")
    print(f"  PyTorch average time: {pytorch_avg:.3f}s")
    print(f"  ONNX average time: {onnx_avg:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  PyTorch FPS: {1.0/pytorch_avg:.2f}")
    print(f"  ONNX FPS: {1.0/onnx_avg:.2f}")
    
    return {
        'pytorch_avg_time': pytorch_avg,
        'onnx_avg_time': onnx_avg,
        'speedup': speedup,
        'pytorch_fps': 1.0/pytorch_avg,
        'onnx_fps': 1.0/onnx_avg
    }

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO model to ONNX format')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model file')
    parser.add_argument('--output', default='src/computer_vision/models', help='Output directory for ONNX model')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Please ensure the YOLO model file exists in the current directory")
        return
    
    # Convert to ONNX
    onnx_path = convert_yolo_to_onnx(args.model, args.output)
    
    if onnx_path and args.benchmark:
        # Run benchmark
        benchmark_models(args.model, onnx_path)
    
    if onnx_path:
        print(f"\n✅ Conversion completed successfully!")
        print(f"  ONNX model: {onnx_path}")
        print(f"  Ready for production use!")
    else:
        print(f"\n❌ Conversion failed!")

if __name__ == "__main__":
    main()
