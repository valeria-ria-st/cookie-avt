import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('runs/detect/cookie_detector-4/weights/best.pt') 

def find_and_click():
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    results = model(screenshot_np)

    clicked_something = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            confidence = box.conf[0]
            class_id = int(box.cls[0]) 

            if confidence > 0.5: 
                if class_id == 0: 
                    pyautogui.click(center_x, center_y)
                    clicked_something = True
                    time.sleep(0.05)
                    
                elif class_id == 1:
                    pyautogui.click(center_x, center_y)
                    clicked_something = True
                    time.sleep(0.5) 

                elif class_id == 2:
                    pyautogui.click(center_x, center_y)
                    clicked_something = True
                    time.sleep(1) 

                elif class_id == 3:
                    pyautogui.click(center_x, center_y)
                    clicked_something = True
                    time.sleep(0.5)

    return clicked_something

while True:
    if find_and_click():
        time.sleep(0.5) 
    else:
        time.sleep(1)   