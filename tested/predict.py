from ultralytics import YOLO

model = YOLO('./pretrained/runs2/detect/train/weights/best.pt')

model.predict('D:/pkm24/counter_vehicle/tested/video.mp4', save=True, show=True, conf=0.6)