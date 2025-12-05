import os
from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import base64
import cv2
import numpy as np
from flask import jsonify


# ---------------------------
#  FLASK APP CONFIG
# ---------------------------
app = Flask(__name__)

#Message
from flask_mail import Mail, Message

# Email settings (GMAIL recommended)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True

# 👉 YOUR EMAIL WHERE YOU WANT TO RECEIVE CONTACT MESSAGES
app.config['MAIL_USERNAME'] = "suryachekuri119@gmail.com"

# 👉 Google App Password (not your login password)
app.config['MAIL_PASSWORD'] = "hqre liec qyye fxlf"

mail = Mail(app)



UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------------------
#  LOAD YOLO MODEL + CSV
# ---------------------------
model = YOLO("model/best.pt")

spares_df = pd.read_csv("spares.csv")

spares_dict = {
    row['defect']: {"part": row['spare_part'], "cost": int(row['cost'])}
    for _, row in spares_df.iterrows()
}

repair_suggestions = {
    "missing_hole": "Re-drill the hole or adjust drilling machine alignment.",
    "spurious_copper": "Remove extra copper using micro-etch or PCB scraping tool.",
    "short": "Remove solder short using soldering iron and flux.",
    "open_circuit": "Repair trace using conductive ink or jumper wire.",
    "mouse_bite": "Clean edges or use solder mask to cover exposed pads."
}

# ---------------------------
#  DETECTION FUNCTION
# ---------------------------
def run_detection(image_path):
    """Runs YOLO detection and returns result filename, defects, repair info, and total cost."""

    results = model.predict(image_path, imgsz=640, conf=0.3, save=True)

    # ---- Save output image ----
    output_filename = "result_" + os.path.basename(image_path)
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    detect_folder = "runs/detect"
    predict_folders = [f for f in os.listdir(detect_folder) if f.startswith("predict")]

    latest_folder = max(
        predict_folders,
        key=lambda f: os.path.getmtime(os.path.join(detect_folder, f))
    )
    latest_path = os.path.join(detect_folder, latest_folder)

    pred_images = sorted(
        os.listdir(latest_path),
        key=lambda x: os.path.getmtime(os.path.join(latest_path, x))
    )

    pred_img_path = os.path.join(latest_path, pred_images[-1])
    Image.open(pred_img_path).save(output_path)

    # ---- Extract detected classes ----
    detected_classes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            detected_classes.append(cls_name)

    detected_classes = list(set(detected_classes))

    if not detected_classes:
        detected_classes = ["No defects detected"]

    # ---- Prepare repair info ----
    repair_info = {}
    total_cost = 0

    for defect in detected_classes:
        suggestion = repair_suggestions.get(defect, "No suggestion available.")
        part = spares_dict.get(defect, {}).get("part", "Not available")
        cost = spares_dict.get(defect, {}).get("cost", 0)

        total_cost += cost

        repair_info[defect] = {
            "suggestion": suggestion,
            "part": part,
            "cost": cost
        }

    return output_filename, detected_classes, repair_info, total_cost


# ---------------------------
#  ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/Home")
def HomePage():
    return render_template("Home.html")

@app.route("/About")
def About():
    return render_template("About.html")

@app.route("/Contact")
def Contact():
    return render_template("Contact.html")

@app.route("/live")
def live_page():
    return render_template("live.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    name = request.form.get("name")
    email = request.form.get("email")
    mobile = request.form.get("mobile")
    message_text = request.form.get("message")

    msg = Message(
        subject=f"📬 New Contact Message from {name}",
        sender=app.config['MAIL_USERNAME'],
        recipients=[app.config['MAIL_USERNAME']]   # YOUR MAIL
    )

    msg.body = f"""
You received a new contact form submission:

Name: {name}
Email: {email}
Mobile: {mobile}

Message:
{message_text}
"""

    mail.send(msg)
    return render_template("Contact.html", success=True)



@app.route("/process_frame", methods=["POST"])
def process_frame():

    # Receive base64 image from frontend
    image_data = request.form["frame"]
    image_bytes = base64.b64decode(image_data.split(",")[1])

    # Convert to OpenCV image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLO on frame
    results = model.predict(frame, imgsz=640, conf=0.3)
    annotated = results[0].plot()

    # Encode annotated frame back to base64
    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_frame = base64.b64encode(buffer).decode("utf-8")

    # Extract detected classes
    defects = []
    for r in results:
        for box in r.boxes:
            name = model.names[int(box.cls[0])]
            defects.append(name)

    defects = list(set(defects))

    return jsonify({
        "frame": "data:image/jpeg;base64," + encoded_frame,
        "defects": defects
    })


@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return "No image uploaded!"

    file = request.files["image"]

    if file.filename == "":
        return "Empty file!"

    # Save uploaded image
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(image_path)

    # Run YOLO
    result_filename, defects, repair_info, total_cost = run_detection(image_path)

    return render_template(
        "result.html",
        result_image=result_filename,
        defects=defects,
        repair_info=repair_info,
        total_cost=total_cost
    )

# ---------------------------
#  START APP
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
