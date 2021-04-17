import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image


pickle_in = open("lr.pkl","rb")
lr=pickle.load(pickle_in)

pickle_in2 = open("cv.pkl","rb")
cv=pickle.load(pickle_in2)

def prediction(normalized):
    prediction1=lr.predict(normalized)
    print(prediction1)
    return prediction1

def normal(text):
    normalized=cv.transform([text])
    print(normalized)
    return normalized

def main():
    st.title("Amazon Review Sentimental Analysis")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text = st.text_input("Input Your Review")
   
    result=""
    if st.button("Predict"):
        normalized=normal(text)
        result=prediction(normalized)
    if(result==1):
        result='Good Review'
    else:
        result='Bad Review'
    st.success(result)

if __name__=='__main__':
    main()