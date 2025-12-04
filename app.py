import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import pandas as pd
import base64
import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------
#  FLASK APP CONFIG
# ---------------------------
app = Flask(__name__)

# Email
from flask_mail import Mail, Message

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = "suryachekuri119@gmail.com"
app.config['MAIL_PASSWORD'] = "hqre liec qyye fxlf"

mail = Mail(app)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ---------------------------
#  LOAD ONNX MODEL + LABELS
# ---------------------------
session = ort.InferenceSession("model/best8.onnx", providers=['CPUExecutionProvider'])

# Load class names (same order as your original model)
model_names = {
    0: "missing_hole",
    1: "spurious_copper",
    2: "short",
    3: "open_circuit",
    4: "mouse_bite"
}

# Load CSV for spares
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
#  ONNX DETECTION HELPERS
# ---------------------------
def preprocess(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    img_resized = cv2.resize(img, (640, 640))
    img_norm = img_resized / 255.0
    img_transposed = img_norm.transpose(2, 0, 1).astype(np.float32)
    input_tensor = img_transposed[np.newaxis, ...]

    return img, input_tensor, (w, h)


def non_max_suppression(pred, iou_thres=0.5, conf_thres=0.3):
    """Very lightweight NMS for YOLO ONNX outputs."""
    pred = pred[pred[:, 4] > conf_thres]
    if len(pred) == 0:
        return []

    boxes = []
    for det in pred:
        x_center, y_center, width, height, conf, cls = det
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes.append([x1, y1, x2, y2, conf, cls])

    return boxes


def draw_boxes(image, detections):
    for x1, y1, x2, y2, conf, cls in detections:
        cls = int(cls)
        name = model_names.get(cls, "unknown")

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, name, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image


# ---------------------------
#  DETECTION FUNCTION
# ---------------------------
def run_detection(image_path):
    """Runs ONNX detection and returns result filename, defects, repair info, and total cost."""

    img, input_tensor, (w, h) = preprocess(image_path)

    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    pred = outputs[0][0]

    detections = non_max_suppression(pred)

    detected_classes = list(set([model_names[int(d[5])] for d in detections]))

    if not detected_classes:
        detected_classes = ["No defects detected"]

    # Draw bounding boxes
    annotated = draw_boxes(img.copy(), detections)

    # Save annotated image
    output_filename = "result_" + os.path.basename(image_path)
    output_path = os.path.join(RESULT_FOLDER, output_filename)
    cv2.imwrite(output_path, annotated)

    # Prepare repair info
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
        recipients=[app.config['MAIL_USERNAME']]
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



# LIVE FRAME PROCESSING
@app.route("/process_frame", methods=["POST"])
def process_frame():
    
    image_data = request.form["frame"]
    image_bytes = base64.b64decode(image_data.split(",")[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Preprocess frame
    resized = cv2.resize(frame, (640, 640))
    norm = resized / 255.0
    transposed = norm.transpose(2, 0, 1).astype(np.float32)
    tensor = transposed[np.newaxis, ...]

    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    pred = outputs[0][0]

    detections = non_max_suppression(pred)
    annotated = draw_boxes(frame.copy(), detections)

    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_frame = base64.b64encode(buffer).decode("utf-8")

    defects = list(set([model_names[int(d[5])] for d in detections]))

    return jsonify({
        "frame": "data:image/jpeg;base64," + encoded_frame,
        "defects": defects
    })


# UPLOAD PAGE
@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return "No image uploaded!"

    file = request.files["image"]

    if file.filename == "":
        return "Empty file!"

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(image_path)

    result_filename, defects, repair_info, total_cost = run_detection(image_path)

    return render_template(
        "result.html",
        result_image=result_filename,
        defects=defects,
        repair_info=repair_info,
        total_cost=total_cost
    )


# START
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
