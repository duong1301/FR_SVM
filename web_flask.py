from flask import Flask, render_template, Response, request
import cv2
from load_save_model import read_model
from numpy import load
import os
import csv
from datetime import datetime
global capture, switch, out

capture = 0
switch = 1

try:
    os.mkdir('./shorts')
except OSError as error:
    pass

img_white_path = os.path.join("static", "image_white")
model = read_model("lbph_svm_model_v2.pkl", path="")
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
read_labels = load("labels.npz")
labels = read_labels['arr_0']
app = Flask(__name__)
camera = cv2.VideoCapture(0)
@app.route("/")
def index():
    user = ["", "", ""]
    return render_template('index.html', user = user)


def predict_image(img_path):
    try:
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = img_gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))
            idx = model.predict([face_img])[0]
        return labels[idx][0]
    except Exception as e:
        pass


def get_infor_user(img_path):
    user_predict = predict_image(img_path)
    infor_user = []
    with open("./inforuser/infor.csv") as f:
        data = csv.reader(f)
        rows = [row for row in data]
        for row in rows:
            if user_predict == row[1]:
                infor_user.append(row[0])
                infor_user.append(row[1])
                return infor_user
    return None

def read_from_webcam():
    global out, capture
    while True:
        success, frame = camera.read()
        if success:
            if capture:
                capture = 0
                p = os.path.sep.join(['shorts', "user_capture.png"])
                cv2.imwrite(p, frame)
            try:

                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass

@app.route("/image_feed")
def image_feed():
    return Response(read_from_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch, camera
    user = ["", "", ""]
    if request.method == 'POST':
        if request.form.get('click') == "Capture":
            global capture
            capture = 1
            if (len(os.listdir(r".\shorts")) != 0):
                user = get_infor_user(r"./shorts\user_capture.png")
                time = str(datetime.now())[:-10]
                print("capture: ", capture)
                print("user: ", user)
                if user is not None:
                    user.append(time)
        elif request.form.get('stop') == "Stop/Start":
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera.VideoCapture(0)
                switch = 1
    elif request.method == 'GET':
        return render_template('index.html', user = user)
    return render_template('index.html', user = user)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
camera.release()
cv2.destroyAllWindows()