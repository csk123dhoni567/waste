from ultralytics import YOLO
import cv2

# Load model
model = YOLO("runs/detect/train-2/weights/best.pt")

cap = cv2.VideoCapture(0)

print("Press SPACE to capture leaf image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    # SPACE key
    if key == 32:
        cv2.imwrite("captured_leaf.jpg", frame)
        print("Image Captured!")

        results = model("captured_leaf.jpg", conf=0.05, save=True)

        found = False

        for r in results:
            for box in r.boxes:
                found = True

                cls = int(box.cls[0])
                disease = model.names[cls]
                confidence = float(box.conf[0]) * 100

                print("\nDisease:", disease)
                print("Confidence:", round(confidence, 2), "%")

        if not found:
            print("No disease detected.")

    # Q key
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()