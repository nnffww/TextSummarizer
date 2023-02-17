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
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Flatten,Dropout,BatchNormalization,Embedding,Input,TimeDistributed, Concatenate, Attention
from tensorflow.keras.utils import plot_model
#from keras.layers import Attention
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
#from attention import AttentionLayer
warnings.filterwarnings("ignore")

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
      data = pd.read_csv(uploaded_file)
      st.dataframe(data)
   if st.button("Summarize"):
      contraction_map = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}
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
        
      def summary_cleaner(text):
        newString = re.sub('"','', text)
        newString = ' '.join([contraction_map[t] if t in contraction_map else t for t in newString.split(" ")])    
        newString = re.sub(r"'s\b","",newString)
        newString = re.sub("[^a-zA-Z]", " ", newString)
        newString = newString.lower()
        tokens=newString.split()
        newString=''
        for i in tokens:
            if len(i)>1:                                 
                newString=newString+i+' '  
        return newString

      #Call the above function
      cleaned_summary = []
      for t in data['Summary']:
        cleaned_summary.append(summary_cleaner(t))

      data['cleaned_text']=cleaned_text
      data['cleaned_summary']=cleaned_summary
      data['cleaned_summary'].replace('', np.nan, inplace=True)
      data.dropna(axis=0,inplace=True)

      # add sostok and eostok at the start and end of summary
      data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x: 'sostok' + ' ' + x + ' ' + 'eostok')

      max_len_text=80 
      max_len_summary=10

      from shutil import copyfile

      copyfile(src="./attention/attention.py", dst="./attention.py")
        
      st.dataframe(data)
       
  
