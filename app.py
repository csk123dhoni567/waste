from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLO model
model = YOLO("runs/detect/train-2/weights/best.pt")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Predict Disease
@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(filepath)

    results = model(filepath, conf=0.001)

    disease = "No Disease Found"
    confidence = 0

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            disease = model.names[cls]

            confidence = float(box.conf[0]) * 100

    return render_template(
        "result.html",
        image=filepath,
        disease=disease,
        confidence=round(confidence, 2)
    )

# Scan Page
@app.route("/scan")
def scan():
    return render_template("scan.html")

# Disease Library
@app.route("/library")
def library():
    return render_template("library.html")

# Expert Advice
@app.route("/expert")
def expert():
    return render_template("expert.html")

# Community
@app.route("/community")
def community():
    return render_template("community.html")

if __name__ == "__main__":
    app.run(debug=True)