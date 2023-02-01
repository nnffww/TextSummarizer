import streamlit as st  
import pandas as pd
import numpy as np
import io
import requests
import re   
import warnings
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.cluster.util import cosine_distance
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Flatten,Dropout,BatchNormalization,Embedding,Input,TimeDistributed
from tensorflow.keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>EXTRACTIVE BASED TEXT SUMMARIZATION USING SENTIMENT ANALYSIS</h1>", unsafe_allow_html=True)
st.markdown("<hr size='5' width='100%;'>", unsafe_allow_html=True)
activities = ["News Article","Summarize"]
choice = st.sidebar.selectbox("Select Activity", activities)

# Primary accent for interactive elements
primaryColor = '#7792E3'

# Background color for the main content area
backgroundColor = '#273346'

# Background color for sidebar and most interactive widgets
secondaryBackgroundColor = '#B9F1C0'

# Color used for almost all text
textColor = '#FFFFFF'

# Font family for all text in the app, except code blocks
# Accepted values (serif | sans serif | monospace) 
# Default: "sans serif"
font = "sans serif"
    
if choice == 'News Article': 
  category = ["Business","Entertaiment","Politics","Sport", "Technology"]
  option = st.selectbox("Select News Article", category)
    
  if option == 'Business':
    url = 'https://raw.githubusercontent.com/faraawaheeda/streamlitProject/main/business%20data.csv?token=GHSAT0AAAAAAB44S4MWPQE3QEME6JT4YSV4Y6Q4XLQ'
    df = pd.read_csv(url,encoding="latin-1")
    st.write(df.head(10))
    st.download_button("Download",
                      df.to_csv(),
                      file_name = 'BusinessArticle.csv',
                      mime = 'text/csv')
                     
if choice == 'Summarize': 
   with st.form(key = 'nlpForm'):
      text = st.text_area("Original Content","Enter text here")
      submitted = st.form_submit_button("Summarize")
      if submitted:
         st.info("Result")
        
   uploaded_txt = st.file_uploader("Choose a file",type=["txt"])
   if uploaded_txt is not None:
      st.write(type(uploaded_txt))
      file_details_txt = {"filename":uploaded_txt.name,"filetype":uploaded_txt.type,"filesize":uploaded_txt.size}
      st.write(file_details_txt)
      if uploaded_txt.type =="text/plain":
         Dftxt = uploaded_txt.read()
         raw_text = str(Dftxt,"utf-8")
         st.text(raw_text)
      if st.button("Summarize"):
        st.write(raw_text)
        st.button("Copy text")
        st.write("Words:")
        
   uploaded_file = st.file_uploader("Choose a file",type=["csv"])
   if uploaded_file is not None:
      st.write("ORIGINAL CONTENT")
      type_file = type(uploaded_file)
      st.write(type_file)
      file_details = {"filename":uploaded_file.name,"filetype":uploaded_file.type,"filesize":uploaded_file.size}
      st.write(file_details)
      df = pd.read_csv(uploaded_file)
      st.dataframe(df)
   if st.button("Summarize"):
      stop_words = set(stopwords.words('english')) 
      def text_cleaner(text):
        newString = text.lower()
        newString = re.sub(r'\([^)]*\)', '', newString)
        newString = re.sub('"','', newString)
        newString = ' '.join([contraction_map[t] if t in contraction_map else t for t in newString.split(" ")])    
        newString = re.sub(r"'s\b","",newString)
        newString = re.sub("[^a-zA-Z]", " ", newString) 
        tokens = [w for w in newString.split() if not w in stop_words]
        long_words=[]
        for i in tokens:
            if len(i)>=3:                  #removing short word
                long_words.append(i)   
        return (" ".join(long_words)).strip()

      cleaned_text = []
      for t in data['Text']:
        cleaned_text.append(text_cleaner(t))
       
  
