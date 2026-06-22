from ultralytics import YOLO
model = YOLO('D:\Codes\fdss\Data\training_runs\chesspiece_yolov83\weights\best.pt')
model.export(format='onnx')
