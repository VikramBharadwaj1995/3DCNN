import cv2, os, torch
import torchvision
import numpy as np

files = os.listdir(".\CharadesEgo")
for file in files:
    frames = []
    cap = cv2.VideoCapture(os.path.join(".\CharadesEgo", file))
    ret, frame = cap.read()
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if count%10 == 0:
                frames.append(frame)
            count += 1
        else:
            break
    filename = os.path.join("downscaled_videos", file)
    torchvision.io.write_video(filename, torch.tensor(np.array(frames)), 15.0)