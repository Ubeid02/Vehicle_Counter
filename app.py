import cv2
import numpy as np
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def resize_frame(frame, target_width=640):
    height, width = frame.shape[:2]
    ratio = target_width / width
    new_height = int(height * ratio)
    return cv2.resize(frame, (target_width, new_height))

def main():
    try:
        # Load Pre-trained YOLOv8 model
        MODEL = 'yolov8n.pt'
        model = YOLO(MODEL)

        # Class names dictionary
        CLASS_NAMES_DICT = model.names
        
        # class_ids of interest - car, motorbike, bus and truck
        CLASS_ID = [2, 3, 5, 7]

        # Video source
        SOURCE_VIDEO_PATH = "tested/video.mp4"  # Ganti dengan path video Anda
        TARGET_VIDEO_PATH = "tested/output/video.mp4"  # Ganti dengan path output yang diinginkan

        # Baca video
        cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
        
        if not cap.isOpened():
            raise Exception(f"Error: Tidak dapat membuka video di {SOURCE_VIDEO_PATH}")

        # Get video properties
        width = 640  # Set target width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video Properties - Width: {width}, Height: {height}, FPS: {fps}")

        # Define counting line
        LINE_START = Point(50, height // 2)
        LINE_END = Point(width - 50, height // 2)

        # Initialize video writer
        out = cv2.VideoWriter(
            TARGET_VIDEO_PATH, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height)
        )

        # Initialize annotators
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1)
        line_counter = LineCounter(start=LINE_START, end=LINE_END)
        line_annotator = LineCounterAnnotator(thickness=2, text_thickness=2, text_scale=1)

        # Dictionary untuk menyimpan jumlah kendaraan
        vehicle_counts = {class_id: 0 for class_id in CLASS_ID}
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"\nSelesai memproses {frame_count} frame")
                break

            frame_count += 1
            print(f"Memproses frame ke-{frame_count}", end='\r')

            # Resize frame
            frame = resize_frame(frame, target_width=width)

            # Detect objects
            results = model(frame)[0]
            
            # Extract detections
            boxes = results.boxes
            detections = Detections(
                xyxy=boxes.xyxy.cpu().numpy(),
                confidence=boxes.conf.cpu().numpy(),
                class_id=boxes.cls.cpu().numpy().astype(int)
            )

            # Filter detections by class
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            # Update line counter
            line_counter.update(detections=detections)

            # Update vehicle counts
            for class_id in CLASS_ID:
                if isinstance(line_counter.in_count, dict):  # Periksa apakah in_count adalah dictionary
                    vehicle_counts[class_id] = line_counter.in_count.get(class_id, 0)

            # Draw annotations
            labels = [
                f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            # Draw detections
            annotated_frame = box_annotator.annotate(
                frame=frame.copy(), 
                detections=detections, 
                labels=labels
            )
            
            # Draw line and counter
            annotated_frame = line_annotator.annotate(
                frame=annotated_frame, 
                line_counter=line_counter
            )

            # Add counter text
            y_pos = height - 20
            for class_id in CLASS_ID:
                class_name = CLASS_NAMES_DICT[class_id]
                count = vehicle_counts[class_id]
                text = f"{class_name}: {count}"
                
                # Tambahkan text ke frame
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y_pos -= 30  # Geser ke atas untuk text berikutnya

            # Write frame to output video
            out.write(annotated_frame)

            # Try to display the frame using cv2.imshow
            try:
                cv2.imshow('Vehicle Detection and Counting', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                # If cv2.imshow fails, try an alternative method using numpy and matplotlib
                try:
                    import matplotlib.pyplot as plt
                    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.pause(0.001)
                    plt.clf()
                except Exception as e:
                    print(f"Tidak dapat menampilkan frame: {str(e)}")

        # Print final counts
        print("\nHasil akhir perhitungan:")
        for class_id in CLASS_ID:
            class_name = CLASS_NAMES_DICT[class_id]
            count = vehicle_counts[class_id]
            print(f"{class_name}: {count}")

    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")
        raise  # Tampilkan stack trace untuk debugging

    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Ignore if windows can't be destroyed
        try:
            plt.close()
        except:
            pass

if __name__ == "__main__":
    main()