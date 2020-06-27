# %%
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import cv2
import time
import playsound


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)

# %%
class Model(object):
    
    EMOTIONS_LIST = ["Angry", "Disgust","Fear", "Happy","Neutral", "Sad","Surprise"]
    audio=["Tracks/Angry.wav","Tracks/Disgust.wav","Tracks/Fear.wav","Tracks/Happy.wav","Tracks/Neutral.wav","Tracks/Sad.wav","Tracks/Surprise.wav"]
    
    def __init__(self, model_json_file, model_weights_file):
        with open("m.json", "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("mw.h5")
    
    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return Model.EMOTIONS_LIST[np.argmax(self.preds)] 
    
    def play(self,img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        a = np.argmax(self.preds)
        playsound.playsound(Model.audio[a],True)
        return;

# %%
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = Model("m.json", "mw.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
now = time.time()
future = now + 5

# %%
while True:
    ret,fr=cap.read()
    if not ret:
        continue
    # returns camera frames along with bounding boxes and predictions
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray_fr, 1.3, 5)
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        
        cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        

    _, jpeg = cv2.imencode('.jpg', fr)
    
    resized_img = cv2.resize(fr, (1000, 700))
    cv2.imshow('Retro',resized_img)
    if time.time() > future:
        if bool(len(faces))==True:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            model.play((roi[np.newaxis, :, :, np.newaxis]))
        future=future+5

    if cv2.waitKey(10) == ord('0'):
        break

cap.release()
cv2.destroyAllWindows


# %%


# %%
