import pickle
import streamlit as st
import keras
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.web.cli as stcli
from keras.src.saving import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
import keras.src.saving


def return_prediction(MODEL, scaler, sample_json):
  temp = sample_json['Celsius']
  val = np.array([[temp]])
  new_val = scaler.transform(val)
  predict_x = MODEL.predict(new_val)
  predict_x_original = scaler.inverse_transform(predict_x)
  Temp_f = (1.8 * predict_x_original) + 32
  return Temp_f
return_prediction(MODEL, scaler, temp_exp)


scaler = pickle.load(open("C:\\Users\\Suyash Pandey\\PycharmProjects\\Temperature_convertor\\SCALER (1).pkl", "rb"))
model = load_model("MODEL.h5")
st.title("Temperature_Convertor")
Celsius = st.number_input('Enter the temperture in celsius')
if st.button('Predict'):
    v = [[Celsius]]
    new_v = scaler.fit_transform(v)
    predict = model.predict(new_v)
    Temp = ((9 * predict)/5) + 32
    st.header(Temp)
