import numpy as np
import random

import matplotlib.pyplot as plt

import cv2
import torch

from ultralytics import YOLO


font = cv2.FONT_HERSHEY_SIMPLEX
colors = dict()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('runs/detect/train6/weights/best.pt')
model.to('cpu')

cam = cv2.VideoCapture(0)

screenshot_idx = 0

while True:
    ret, frame = cam.read()
    result = model.predict([frame], conf=0.5, imgsz=640)[0]
    boxes = result.boxes.xyxy.to(torch.int32).to('cpu')

    probs = result.probs
    print(result.boxes)

    for i in range(len(boxes)):
        nboxes = boxes[i].numpy()
        print(nboxes)
        center_x = nboxes[0]
        center_y = nboxes[1]
        width = nboxes[2]
        height = nboxes[3]
        x = center_x - width
        y = center_y - height
        frame = cv2.rectangle(frame, (center_x, center_y), (width, height), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    key = cv2.waitKey(1)
    if key == 32:
        cv2.imwrite(f'frame_{screenshot_idx}.png', frame)
        screenshot_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()
