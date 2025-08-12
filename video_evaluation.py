import cv2
from deep_sort_tracker import FeatureExtractor, EmbeddingBuffer, match_detections_to_tracks
import torch
import numpy as np
from ultralytics import YOLO

yolo = YOLO("yolov8m.pt")  


device=torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
extractor = FeatureExtractor("your/trained/model/path", device)
buffer = EmbeddingBuffer()


cap = cv2.VideoCapture("your/video/path")
if not cap.isOpened():
    print("Could not open video.mp4")
    exit()


width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

track_history = {}
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)[0]
    detections = []

    for i, det in enumerate(results.boxes.data):
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if int(cls) not in [2, 7]: 
            continue
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        detections.append((i, crop))

    track_ids = list(buffer.buffer.keys())
    matched = match_detections_to_tracks(detections, track_ids, extractor, buffer)

  
    for i, track_id in matched:
        x1, y1, x2, y2, conf, cls = results.boxes.data[i].cpu().numpy()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((frame_index, (x1, y1, x2, y2)))

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("🎥 Output saved as output.mp4")