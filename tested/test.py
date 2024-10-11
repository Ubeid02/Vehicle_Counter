from ultralytics import YOLO
import cv2

# Initialize the YOLO model with your weights file
model = YOLO('./pretrained/runs/detect/train/weights/best.pt')

# Start choose capturing video from the camera
cap = cv2.VideoCapture('D:/pkm24/counter_vehicle/tested/video.mp4')

while True:
    # Read a frame from the camera
    _, frame = cap.read()
    
    # Perform object detection on the frame
    results = model.predict(frame)
    
    # Iterate through the detection results
    for result in results:
        # Get the bounding box coordinates in (top, left, bottom, right) format
        boxes = result.boxes.xyxy
        # Get the confidence scores
        conf = result.boxes.conf
        # Get the class labels
        cls = result.boxes.cls
        
        # Process the bounding box coordinates, confidence scores, and class labels
        # For example, you can print them or draw bounding boxes on the frame
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            print(f"Bounding box coordinates: {x1}, {y1}, {x2}, {y2}")
            print(f"Confidence score: {conf[i]}")
            print(f"Class label: {model.names[int(cls[i])]}")
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,  255,  0),  2)
            # Draw the class label on the frame
            cv2.putText(frame, model.names[int(cls[i])], (int(x1), int(y1 -  5)),
                        cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,  255,  0),  2)
    
    # Display the frame with bounding boxes
    cv2.imshow('YOLO Detection', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
