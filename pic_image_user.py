from time import sleep

import cv2
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

count = 71
leap = 1
data_path = "dataset/"
username_capturing = input("Enter your name: ")
user_path = os.path.join(data_path, username_capturing)
os.mkdir(user_path)
os.chdir(user_path)
while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if isSuccess & leap % 2:
        sleep(0.1)
        file_name =str(username_capturing +'{}.jpg'.format(str(datetime.now())[:-7].replace(':', '-').replace(' ', '-') + str(count)))
        cv2.imwrite(file_name, frame)
        count -= 1
    leap += 1
    cv2.imshow("Face Capturing", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()