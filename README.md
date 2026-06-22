# Real-Time Chess Piece Detection

A YOLOv8-based computer vision system for real-time detection and classification of chess pieces from live video streams.

## Technologies

- Python
- OpenCV
- YOLOv8
- Roboflow

## Results

- mAP@50: 0.99
- mAP@50-95: 0.80
- Precision: 0.98
- Recall: 0.99

## Features

- Real-time chess piece detection
- 13 chess piece classes
- Live webcam inference
- FEN generation support
- ONNX export support

## Training Details

- ChessRed2K dataset (10,800 images)
- 60 training epochs
- Early stopping
- Mosaic augmentation
- Brightness augmentation
