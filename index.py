from ultralytics import YOLO
import cv2
import math
import cvzone
from supervision.geometry.dataclasses import Point
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.geometry.dataclasses import Rect

cap = cv2.VideoCapture("tested/video.mp4")

model = YOLO("pretrained/runs2/detect/train/weights/best.pt")

CLASS_NAMES_DICT = model.model.names

# Define the color for annotation (for example, green color)
annotation_color = Rect(0, 255, 0)  # Green color in BGR format

# Menentukan garis horizontal di tengah frame
line_start = Point(50, 1500)
line_end = Point(3840-50, 1500)
line_counter = LineCounter(line_start, line_end)
line_counter_annotator = LineCounterAnnotator(line_start, line_end, color=annotation_color)

while True:
    success, img = cap.read()

    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Boundeing box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)
            w, h = x2-x1, y2-y1

            # confidence
            conf = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])
            currentClass = CLASS_NAMES_DICT[cls]

            if currentClass == "bus" or currentClass == "cars" or currentClass == "motorbike" or currentClass == "truck" and conf > 5:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=5)
                cvzone.cornerRect(img, (x1, y1, w, h), l=8)

                # Menghitung kendaraan yang melewati garis
                line_counter.update([(box.xyxy, conf, cls, None)])

                # Menambahkan visualisasi dan informasi kendaraan
                line_counter_annotator.annotate(img, line_counter)

    # Menghitung total jumlah kendaraan yang melewati garis
    total_vehicles = line_counter.in_count + line_counter.out_count

    # Menampilkan jumlah total kendaraan yang melewati garis
    cvzone.putTextRect(img, f'Vehicles: {total_vehicles}', (10, 30), scale=1, thickness=2, offset=5)

    cv2.imshow("Vehicle Count", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Tekan 'q' untuk keluar dari loop
        break

cv2.destroyAllWindows()