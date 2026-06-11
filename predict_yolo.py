from ultralytics import YOLO

# Treatment suggestions
treatments = {
    "Corn leaf blight": "Use fungicide and remove infected leaves.",
    "Tomato leaf late blight": "Apply copper fungicide.",
    "Apple Scab Leaf": "Use resistant varieties and fungicides."
}

# Load trained model
model = YOLO("runs/detect/train-2/weights/best.pt")

# Ask user for image path
img_path = input("Enter image path: ")

# Run prediction
results = model(img_path, conf=0.05, save=True)

found = False

# Process results
for r in results:
    for box in r.boxes:

        found = True

        cls = int(box.cls[0])
        disease = model.names[cls]
        confidence = float(box.conf[0]) * 100

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print("\nPrediction Result")
        print("-----------------")
        print("Disease:", disease)
        print("Confidence:", round(confidence, 2), "%")
        print("Location:", [round(x1, 2), round(y1, 2),
                             round(x2, 2), round(y2, 2)])

        # Show treatment
        if disease in treatments:
            print("Treatment:", treatments[disease])
        else:
            print("Treatment: Not available")

if not found:
    print("\nNo disease detected.")