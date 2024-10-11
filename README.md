# Vehicle Counting Using YOLO Object Detection

This project aims to create a system that can count vehicles passing through a defined area using **YOLO (You Only Look Once)** object detection. The system uses pre-trained YOLO models to detect different vehicle types such as cars, motorbikes, buses, and trucks in video footage. Although this project is still a work in progress, the system successfully detects vehicles but is not yet fully accurate in counting vehicles that pass through the designated line.

## 🚗 Project Features

- **Object Detection with YOLO**: Utilizes the YOLOv8 model for real-time object detection.
- **Vehicle Counting**: Detects and attempts to count cars, motorbikes, buses, and trucks in the video.
- **Real-time Processing**: Processes video frames in real-time, displaying detected vehicles with bounding boxes.
- **Visualization**: Shows the line counter and vehicle count on the processed video output.

## ❗ Current Issues

- The system is not yet fully able to **accurately count vehicles** that pass through the designated area.
- Further improvements and bug fixes are required to make the counting more reliable.

## 🛠️ Technologies Used

- **YOLOv8**: Pre-trained object detection model for detecting vehicles.
- **OpenCV**: Used for video processing and displaying bounding boxes.
- **NumPy**: Used for matrix operations and filtering detections.

## 📹 Video Example

The system processes video frames from a source video, detects vehicles, and attempts to count them as they pass a defined line in the video.

## ⚙️ How it Works

1. **YOLOv8 Model**: The system loads a pre-trained YOLOv8 model to detect vehicle types such as cars, motorbikes, buses, and trucks.
2. **Counting Line**: A horizontal line is drawn across the video frame, and vehicles passing this line are counted.
3. **Bounding Boxes and Labels**: Detected vehicles are displayed with bounding boxes and class labels (e.g., "Car", "Truck").
4. **Real-time Processing**: Each frame of the video is processed in real-time to update the vehicle count.

## 🚧 Status

The project is **not yet 100% complete**. While object detection works, the vehicle counting feature still needs improvement. The system struggles to consistently count vehicles as they pass the line.

## 📦 Requirements

## 🔧 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ubeid02/Vehicle_Counter.git

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt

3. Modify the video source and output paths in the code:

    ```bash
    SOURCE_VIDEO_PATH = "tested/video.mp4"  # Replace with your input video path
    TARGET_VIDEO_PATH = "tested/output/video.mp4"  # Replace with your desired output path

4. Run the application:

    ```bash
    app.py

The system will process the video, display the detection results, and attempt to count vehicles as they pass the counting line.

📦 Requirements
Make sure you have the following libraries installed:

```bash
pip install torch==1.13.0
pip install ultralytics==8.0.196
pip install supervision==0.18.0
pip install onemetric
pip install opencv-python
