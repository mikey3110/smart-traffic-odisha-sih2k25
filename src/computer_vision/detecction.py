from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model
# 'yolov8n.pt' is a good starting point as it's small and fast
model = YOLO('yolov8n.pt')

# Path to your video file
video_path = 'data/videos/vehicles.mp4'  # Make sure to change this to your video filename

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Loop through the video frames
while True:
    # Read a frame
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Perform object detection on the frame
    # The 'classes' parameter can be used to filter for specific objects.
    # YOLO COCO dataset class indices for vehicles:
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    results = model(frame, classes=[2, 3, 5, 7])

    # Get the processed frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('Vehicle Detection', annotated_frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
