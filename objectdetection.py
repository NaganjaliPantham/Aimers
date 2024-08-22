import cv2
import numpy as np
from ultralytics import YOLO

def load_model(model_path='yolov8n.pt'):
    # Load a pre-trained YOLOv8 model
    model = YOLO(model_path)
    return model

def detect_objects(model, image_path):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Parse results
    predictions = results.pandas().xyxy[0]  # pandas DataFrame with bounding boxes

    return img, predictions

def draw_boxes(img, predictions):
    # Draw bounding boxes on the image
    for i, row in predictions.iterrows():
        x1, y1, x2, y2, conf, cls = row[:6]
        label = f"{row['name']} {conf:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

def main():
    model_path = 'yolov8n.pt'  # You can use other models like 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    image_path = 'path/to/your/image.jpg'

    model = load_model(model_path)
    img, predictions = detect_objects(model, image_path)
    img_with_boxes = draw_boxes(img, predictions)

    # Show the result
    cv2.imshow('Object Detection', img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
