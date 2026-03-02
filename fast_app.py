from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import time
import json
import base64
import uvicorn

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLO("runs/detect/train/weights/best.pt")


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    raw_size_mb = len(contents) / (1024 * 1024)

    # Inference
    start = time.time()
    results = model.predict(img, conf=0.25, verbose=False)
    inference_time_ms = (time.time() - start) * 1000

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            detections.append({
                "object": "ship",
                "confidence": round(conf, 4),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f'{det["object"]} {det["confidence"]}',
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # Encode annotated image
    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    metadata_json = json.dumps(detections)
    metadata_size_kb = len(metadata_json.encode()) / 1024

    bandwidth_saved = (
        100 - ((metadata_size_kb / 1024) / raw_size_mb * 100)
        if raw_size_mb > 0 else 0
    )

    return {
        "inference_time_ms": round(inference_time_ms, 2),
        "raw_image_MB": round(raw_size_mb, 4),
        "metadata_KB": round(metadata_size_kb, 4),
        "bandwidth_saved_percent": round(bandwidth_saved, 4),
        "detections": detections,
        "annotated_image": img_base64
    }


if __name__ == "__main__":
    uvicorn.run("fast_app:app", host="127.0.0.1", port=8000, reload=False)