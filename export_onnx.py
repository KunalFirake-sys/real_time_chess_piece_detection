import cv2

cap = cv2.VideoCapture(0)
print(f"Camera opened? {cap.isOpened()}")

while True:
    ret, frame = cap.read()
    print(f"Frame read: {ret}")
    if not ret:
        break
    cv2.imshow("Test Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
