from ultralytics import YOLO
import argparse

def export_yolo_to_onnx(weights_path, output_dir="src/cv/models/"):
    model = YOLO(weights_path)
    model.export(format="onnx", dynamic=True, opset=12, simplify=True, imgsz=640, half=True)
    print(f"✅ Exported {weights_path} to ONNX in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to YOLO weights (.pt)")
    args = parser.parse_args()
    export_yolo_to_onnx(args.weights)
