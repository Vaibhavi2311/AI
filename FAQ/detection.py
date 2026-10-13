
import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check webcam availability
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Object Detection and Tracking Started")
print("Press 'q' to quit")

while True:
    # Read video frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read frame")
        break

    # Detect and track objects using ByteTrack
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False
    )

    # Draw bounding boxes, labels, and tracking IDs
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow(
        "Object Detection and Tracking",
        annotated_frame
    )

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

