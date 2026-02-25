import cv2
from ultralytics import YOLO

# Load YOLOv8 pretrained model (auto-downloads if not present)
model = YOLO("yolov8n.pt")

# Read image (make sure path is correct)
img = cv2.imread(r"C:\Users\mohur\Downloads\cars.jpg")

if img is None:
    print("Error: Image not found.")
    exit()

# Run YOLO detection
results = model(img)

# Draw bounding boxes
annotated_img = results[0].plot()

# Show output
cv2.imshow("YOLOv8 Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()