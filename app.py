import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras import models
from cv2 import imread

model=models.load_model('Facial_Expression_classification_CNN_LSTM.keras')
emotions=[['angry'],
 ['disgust'],
 ['fear'],
 ['happy'],
 ['neutral'],
 ['sad'],
 ['surprise']]

st.header("Facial Expression Detection")
image_path=st.text_input("Enter Image Path")

image=cv2.imread(image_path)[:,:,0]
image=cv2.resize(image,(48,48))
image=np.invert(np.array([image]))

output=np.argmax(model.predict(image))
outcome=emotions[output]
stn='Emotion is '+ str(outcome[0])
st.markdown(stn)

image_name = os.path.basename(image_path)
st.image('Google_Images/' + image_name, width=300)