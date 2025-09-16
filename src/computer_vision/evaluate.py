import os
import subprocess

report_file = "docs/cv_report.md"

def run_validation():
    cmd = [
        "yolo", "val",
        "model=src/cv/models/best.onnx",
        "data=data/cv_test.yaml",
        "imgsz=640"
    ]
    subprocess.run(cmd)

def collect_results():
    results_path = "runs/detect/val/weights/results.csv"
    if not os.path.exists(results_path):
        print("⚠ Results not found. Did YOLO val run?")
        return None
    with open(results_path, "r") as f:
        lines = f.readlines()
    return lines[-1]  # last metrics row

def update_report(metrics):
    os.makedirs("docs", exist_ok=True)
    with open(report_file, "a") as f:
        f.write("\n## Accuracy Validation Results (Day-1)\n")
        f.write("| Precision | Recall | mAP50 |\n")
        f.write("|-----------|--------|-------|\n")
        cols = metrics.strip().split(",")
        prec, recall, map50 = cols[2], cols[3], cols[4]
        f.write(f"| {prec} | {recall} | {map50} |\n")
    print(f"✅ Report updated at {report_file}")

if __name__ == "__main__":
    run_validation()
    metrics = collect_results()
    if metrics:
        update_report(metrics)
