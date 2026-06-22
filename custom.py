import cv2
import os

# Create directories
os.makedirs('webcam_training_data/images', exist_ok=True)
os.makedirs('webcam_training_data/labels', exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press SPACE to capture frame, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (640, 640))
    cv2.imshow('Capture Frames', frame_resized)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # Space to capture
        img_path = f'webcam_training_data/images/webcam_{count:04d}.jpg'
        cv2.imwrite(img_path, frame_resized)
        print(f"✅ Saved {img_path}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} frames")
