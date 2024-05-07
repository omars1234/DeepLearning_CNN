import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

model=pickle.load(open("ImageClassification_model.pkl","rb"))
Tumor_types=['meningioma','glioma',  'notumor', 'pituitary']


def show_predict_page():
    st.title("Tumor image Classification")
    st.write("Tumer Types are : glioma, meningioma, notumor, pituitary ")


    image_upload=st.file_uploader("Insert Tumor Image",type="jpg")

    if image_upload is not None:

        file_bytes = np.asarray(bytearray(image_upload.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.sidebar.image(opencv_image)

        submit=st.button("To Predict the Tumor Type click her")

        if submit:
              resize=tf.image.resize(opencv_image,(256,256))

              make_prediction=model.predict(np.expand_dims(resize/255,0))
              

              st.write(Tumor_types,make_prediction)    


              

              #for i in make_prediction:
               #    for x in Tumor_types:
                #       st.write(x,i)






if __name__=="__main__" :
    show_predict_page()