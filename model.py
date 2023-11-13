from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from  numpy import load
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lpbh_svm import LPBH_SVM_Recognize
# from skimage.future import local_binary_pattern
from numpy import savez_compressed

def show_images(images_class, label):
    plt.figure(figsize=(14, 5))
    k = 0
    for i in range(1, 6):
        plt.subplot(1, 5, i)
        try:
            plt.imshow(images_class[k][:,:,::-1])
        except:
            plt.imshow(images_class[k], cmap='gray')
        plt.title(label)
        plt.axis('off')
        plt.tight_layout()
        k += 1
    plt.show()


data_folder = "dataset/"
names = []
images = []
for folder in os.listdir(data_folder):
    user_path = os.path.join(data_folder, folder)
    for file_name in os.listdir(user_path):
        if file_name.find(".jpg") > -1:
            img = cv2.imread(os.path.join(user_path, file_name))
            images.append(img)
            names.append(folder)

labels = np.unique(names)
savez_compressed("labels.npz", labels)
labels = load("labels.npz")

# for label in labels:
#     idx = np.where(label == np.array(names))[0]
#     images_class = images[idx[0]:idx[-1] + 1]
#     show_images(images_class, label)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img, idx):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    try:
        x, y, w, h = faces[0]
        img = img[y:y + h, x:x + w]
        img = cv2.resize(img, (100, 100))
    except:
        img = None
    return img
croped_images = []

index_of_loss = []
for i, img in enumerate(images):
    img = detect_face(img, i)
    if img is not None:
        croped_images.append(img)
    else:
        index_of_loss.append(i)
count = 0
for i in index_of_loss:
    i -= count
    names.remove(names[i])
    count += 1
# for label in labels:
#     idx = np.where(label == np.array(names))[0]
#     img_class = croped_images[idx[0]:idx[-1] + 1]
#     show_images(img_class, label)

encoder_label = LabelEncoder()
encoder_label.fit(names)
labels_encoder = encoder_label.transform(names)


x_train, x_test, y_train, y_test = train_test_split(np.array(croped_images, dtype=np.float32), np.array(labels_encoder), test_size=0.2, random_state=42)
lpbh_svm_model = LPBH_SVM_Recognize()
lpbh_svm_model.train(x_train, y_train)
#
from load_save_model import save_model, read_model
save_model(lpbh_svm_model, "lbph_svm_model_v2.pkl", path="")
lpbh_svm_model = read_model("lbph_svm_model_v2.pkl", path="")
y_predict = lpbh_svm_model.predict(x_test)[0]
# print("== Classification Report - SVM + LBPH Scikit ==\n")
# print(classification_report(y_test, y_predict, target_names=labels))

