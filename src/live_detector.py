import cv2
import pickle
import numpy as np
from skimage.transform import resize
from config import *

# Load model and spots
model = pickle.load(open('../models/model.p', 'rb'))
spots = pickle.load(open('../data/spots.p', 'rb'))

video = cv2.VideoCapture('../data/videos/parking_1920_1080.mp4')

frame_count = 0
spot_colors = [(0, 0, 255)] * len(spots)
spots_per_frame = 5  # process 5 spots per frame

while True:
    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # process a small batch of spots each frame
    start_idx = (frame_count * spots_per_frame) % len(spots)
    end_idx = min(start_idx + spots_per_frame, len(spots))

    for i in range(start_idx, end_idx):
        x, y, w, h = spots[i]
        if w <= 0 or h <= 0:
            continue
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            continue

        spot_crop = frame[y:y+h, x:x+w]
        spot_resized = resize(spot_crop, IMAGE_SIZE)
        spot_resized = spot_resized / 255.0
        spot_flat = spot_resized.flatten().reshape(1, -1)
        prediction = model.predict(spot_flat)
        spot_colors[i] = (0, 255, 0) if prediction == 0 else (0, 0, 255)

    # always draw all rectangles
    empty_count = 0
    for i, (x, y, w, h) in enumerate(spots):
        cv2.rectangle(frame, (x, y), (x+w, y+h), spot_colors[i], 2)
        if spot_colors[i] == (0, 255, 0):
            empty_count += 1

    # show count
    cv2.rectangle(frame, (0, 0), (250, 40), (0, 0, 0), -1)
    cv2.putText(frame, 'Empty: {}/{}'.format(empty_count, len(spots)),
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Parking Detector', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    frame_count += 1

video.release()
cv2.destroyAllWindows()