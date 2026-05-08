"""
PPE Guard — Flask Backend
Run: python app.py
Then open: http://localhost:5000
"""

import cv2
import numpy as np
import base64
import threading
import time
from collections import deque
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "hard_hat": "best.pt",
    "vest":     "vest_best.pt",
    "gloves":   "gloves_best.pt",
    "shoes":    "shoes_best.pt",
    "goggles":  "goggles_best.pt",
}
POSITIVE_CLASS = {"hard_hat": 0, "vest": 1, "gloves": 1, "shoes": 1, "goggles": 1}
REGIONS = {
    "hard_hat": (0.00, 0.30), "goggles": (0.02, 0.28),
    "vest":     (0.22, 0.65), "gloves":  (0.45, 0.92),
    "shoes":    (0.78, 1.00),
}
LABELS = {
    "hard_hat": "Hard Hat",  "vest":    "Safety Vest",
    "gloves":   "Gloves",    "shoes":   "Safety Shoes",
    "goggles":  "Goggles",
}
CONF_MAP = {
    "hard_hat": 0.25, "vest": 0.40,
    "gloves":   0.40, "shoes": 0.40, "goggles": 0.75,
}
SMOOTH_WIN  = 15
VOTE_PCT    = 30
PERSON_CONF = 0.40

# ── Global state ───────────────────────────────────────────────────────────────
models_loaded  = False
person_model   = None
ppe_models     = {}
smooth_buffers = {k: deque(maxlen=SMOOTH_WIN) for k in LABELS}

webcam_active  = False
webcam_thread  = None
latest_frame   = None
frame_lock     = threading.Lock()

detection_state = {k: "waiting" for k in LABELS}
session_stats   = {"frames": 0, "persons": 0, "fps": 0.0,
                   "compliant": 0, "violations": 0}

def load_models():
    global models_loaded, person_model, ppe_models
    from ultralytics import YOLO
    try:
        person_model = YOLO("yolov8s.pt")
        ppe_models   = {k: YOLO(v) for k, v in MODEL_PATHS.items()}
        models_loaded = True
        print("✓ All models loaded")
    except Exception as e:
        print(f"✗ Model load error: {e}")
        models_loaded = False

def smooth(key, det):
    smooth_buffers[key].append(det)
    votes = sum(smooth_buffers[key])
    return votes >= max(1, len(smooth_buffers[key]) * VOTE_PCT // 100)

def process_frame(frame):
    global detection_state, session_stats
    fh, fw = frame.shape[:2]
    persons = person_model(frame, classes=[0], conf=PERSON_CONF, verbose=False)
    n_p     = len(persons[0].boxes)
    fs      = {k: "waiting" for k in LABELS}

    for box in persons[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ph      = y2 - y1
        is_full = ph > fh * 0.50
        cv2.rectangle(frame, (x1,y1), (x2,y2), (60,80,110), 1)

        for key, model in ppe_models.items():
            tp, bp  = REGIONS[key]
            cy1 = max(0,  y1+int(ph*tp)-10)
            cy2 = min(fh, y1+int(ph*bp)+10)
            cx1 = max(0,  x1-5)
            cx2 = min(fw, x2+5)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0: continue
            if not is_full and key in ["vest","gloves","shoes"]: continue

            res = model(crop, conf=CONF_MAP[key], verbose=False)
            raw = any(int(b.cls) == POSITIVE_CLASS[key] for b in res[0].boxes)
            det = smooth(key, raw)
            fs[key] = "ok" if det else "missing"
            col = (46,204,113) if det else (231,76,60)

            if res[0].boxes:
                for b in res[0].boxes:
                    bx1,by1,bx2,by2 = map(int, b.xyxy[0])
                    bx1+=cx1; bx2+=cx1; by1+=cy1; by2+=cy1
                    cv2.rectangle(frame,(bx1,by1),(bx2,by2),col,2)
                    lbl = f"{LABELS[key]}: {'OK' if det else 'MISS'}"
                    (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
                    cv2.rectangle(frame,(bx1,by1-th-6),(bx1+tw+8,by1),col,-1)
                    cv2.putText(frame,lbl,(bx1+4,by1-4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.circle(frame,((x1+x2)//2,(cy1+cy2)//2),4,col,-1)

    detection_state = fs
    session_stats["persons"] = n_p
    session_stats["frames"] += 1

    all_ok = all(v=="ok" for v in fs.values() if v!="waiting")
    if any(v!="waiting" for v in fs.values()):
        if all_ok: session_stats["compliant"]  += 1
        else:      session_stats["violations"] += 1

    return frame

def webcam_worker():
    global latest_frame, webcam_active
    cap = cv2.VideoCapture(0)
    t0  = time.time()
    cnt = 0
    while webcam_active:
        ret, frame = cap.read()
        if not ret: break
        cnt += 1
        if models_loaded:
            frame = process_frame(frame)
        session_stats["fps"] = cnt / (time.time()-t0+1e-9)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buf.tobytes()
    cap.release()
    webcam_active = False

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "models_loaded":    models_loaded,
        "detection_state":  detection_state,
        "session_stats":    session_stats,
        "webcam_active":    webcam_active,
    })

@app.route("/api/webcam/start", methods=["POST"])
def webcam_start():
    global webcam_active, webcam_thread, smooth_buffers
    if not webcam_active:
        webcam_active  = True
        smooth_buffers = {k: deque(maxlen=SMOOTH_WIN) for k in LABELS}
        webcam_thread  = threading.Thread(target=webcam_worker, daemon=True)
        webcam_thread.start()
    return jsonify({"ok": True})

@app.route("/api/webcam/stop", methods=["POST"])
def webcam_stop():
    global webcam_active
    webcam_active = False
    return jsonify({"ok": True})

@app.route("/api/webcam/frame")
def webcam_frame():
    with frame_lock:
        f = latest_frame
    if f is None:
        return jsonify({"frame": None})
    return jsonify({"frame": base64.b64encode(f).decode()})

@app.route("/api/analyse", methods=["POST"])
def analyse_image():
    global smooth_buffers
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 500
    file  = request.files.get("image")
    if not file:
        return jsonify({"error": "No image"}), 400
    arr   = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    smooth_buffers = {k: deque(maxlen=1) for k in LABELS}
    frame = process_frame(frame)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jsonify({
        "frame":            base64.b64encode(buf.tobytes()).decode(),
        "detection_state":  detection_state,
    })

@app.route("/api/settings", methods=["POST"])
def update_settings():
    global CONF_MAP, PERSON_CONF, SMOOTH_WIN, VOTE_PCT, smooth_buffers
    data = request.json
    PERSON_CONF         = float(data.get("person_conf",  PERSON_CONF))
    CONF_MAP["hard_hat"]= float(data.get("hh_conf",      CONF_MAP["hard_hat"]))
    CONF_MAP["vest"]    = float(data.get("v_conf",       CONF_MAP["vest"]))
    CONF_MAP["gloves"]  = float(data.get("g_conf",       CONF_MAP["gloves"]))
    CONF_MAP["shoes"]   = float(data.get("s_conf",       CONF_MAP["shoes"]))
    CONF_MAP["goggles"] = float(data.get("go_conf",      CONF_MAP["goggles"]))
    SMOOTH_WIN          = int(data.get("smooth_win",     SMOOTH_WIN))
    VOTE_PCT            = int(data.get("vote_pct",       VOTE_PCT))
    smooth_buffers      = {k: deque(maxlen=SMOOTH_WIN) for k in LABELS}
    return jsonify({"ok": True})

@app.route("/api/reset", methods=["POST"])
def reset_stats():
    global session_stats, detection_state, smooth_buffers
    session_stats   = {"frames":0,"persons":0,"fps":0.0,"compliant":0,"violations":0}
    detection_state = {k: "waiting" for k in LABELS}
    smooth_buffers  = {k: deque(maxlen=SMOOTH_WIN) for k in LABELS}
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("Loading models...")
    load_models()
    print("Starting PPE Guard on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
