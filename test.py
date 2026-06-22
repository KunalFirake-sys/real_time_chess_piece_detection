from ultralytics import YOLO
import cv2

# === CONFIGURE PATHS ===
MODEL_PATH = r"D:\Codes\fdss\finetuned_training_runs\chesspiece_finetuned_aggaug\weights\best.pt"  # path to your trained .pt model
IMAGE_PATH = r"D:\Codes\fdss\Data\test\images\8ff752f9ed443e6e49d495abfceb2032_jpg.rf.c3e91277eea99c26328e39a6f0285189.jpg" # input image to test
SAVE_PATH = r"D:\Codes\fdss\Data\prediction.jpg"   # where to save visualized prediction

# Class names (update if your classes differ)
CLASS_NAMES = ['bishop', 'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen',
               'black-rook', 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']


def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Run prediction
    results = model(IMAGE_PATH)

    # Get OpenCV image for annotation
    img = cv2.imread(IMAGE_PATH)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
            cv2.rectangle(img, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), (0,255,0), -1)
            cv2.putText(img, label, (x1, y1-2), 0, 0.5, (0,0,0), 1, lineType=cv2.LINE_AA)

    # Save and show
    cv2.imwrite(SAVE_PATH, img)
    print("Prediction saved to:", SAVE_PATH)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
