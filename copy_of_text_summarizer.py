import streamlit as st 
import numpy as np  
import pandas as pd 
import io
import requests
import re  
import os          
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import defaultdict
from pathlib import Path
import warnings
from attention import AttentionLayer
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Extractive Text Summarization", page_icon=":tada:", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>EXTRACTIVE BASED TEXT SUMMARIZATION USING SENTIMENT ANALYSIS</h1>", unsafe_allow_html=True)
st.markdown("<hr size='5' width='100%;'>", unsafe_allow_html=True)
activities = ["News Article","Summarize"]
choice = st.sidebar.selectbox("Select Activity", activities)

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

    st.write(df)

    import nltk
    nltk.download('stopwords')

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
    for t in df['Text']:
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
    for t in df['Summary']:
      cleaned_summary.append(summary_cleaner(t))

    df['cleaned_text']=cleaned_text
    df['cleaned_summary']=cleaned_summary
    df['cleaned_summary'].replace('', np.nan, inplace=True)
    df.dropna(axis=0,inplace=True)

    # add sostok and eostok at the start and end of summary
    df['cleaned_summary'] = df['cleaned_summary'].apply(lambda x: 'sostok' + ' ' + x + ' ' + 'eostok')

    max_len_text=80 
    max_len_summary=10

    st.write(df)

    from shutil import copyfile
    copyfile(src = "./attention/attention.py", dst = "./attention.py")


    pd.set_option("display.max_colwidth", 200)
    warnings.filterwarnings("ignore")

    from sklearn.model_selection import train_test_split
    x_tr,x_val,y_tr,y_val=train_test_split(df['cleaned_text'],df['cleaned_summary'],test_size=0.1,random_state=0,shuffle=True)

    #prepare a tokenizer for reviews on training data

    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_tr))

    #convert text sequences into integer sequences
    x_tr    =   x_tokenizer.texts_to_sequences(x_tr) 
    x_val   =   x_tokenizer.texts_to_sequences(x_val)

    #padding zero upto maximum length
    x_tr    =   pad_sequences(x_tr,  maxlen=max_len_text, padding='post') 
    x_val   =   pad_sequences(x_val, maxlen=max_len_text, padding='post')

    x_voc_size   =  len(x_tokenizer.word_index) +1

    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_tr))

    #convert summary sequences into integer sequences
    y_tr    =   y_tokenizer.texts_to_sequences(y_tr) 
    y_val   =   y_tokenizer.texts_to_sequences(y_val) 

    #padding zero upto maximum length
    y_tr    =   pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
    y_val   =   pad_sequences(y_val, maxlen=max_len_summary, padding='post')

    y_voc_size  =   len(y_tokenizer.word_index) +1
 
    from keras import backend as K 
    K.clear_session() 
    latent_dim = 500 
 
    # Encoder 
    encoder_inputs = Input(shape=(max_len_text,)) 
    enc_emb = Embedding(x_voc_size, latent_dim,trainable=True)(encoder_inputs) 

    #Preparing LSTM layer 1 
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

    #Preparing LSTM layer 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

    #Preparing LSTM layer 3
    encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

    # Decoder layer 
    decoder_inputs = Input(shape=(None,)) 
    dec_emb_layer = Embedding(y_voc_size, latent_dim,trainable=True) 
    dec_emb = dec_emb_layer(decoder_inputs) 


    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c]) 

    #Dense layer
    # decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax')) 
    #decoder_outputs = decoder_dense(decoder_concat_input)

    #Preparing the Attention Layer
    attn_layer = AttentionLayer(name='attention_layer') 
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])\

    #Adding the dense layer
    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax')) 
    decoder_outputs = decoder_dense(decoder_concat_input) 

    # Prepare the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.summary()

    # Compiling the RNN model
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index

    # # Inference Models

    # Encode the input sequence to get the feature vector
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_len_text,latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb2= dec_emb_layer(decoder_inputs) 
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 

    # Final decoder model
    decoder_model = Model(
      [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
      [decoder_outputs2] + [state_h2, state_c2])

    def decode_sequence(inputText):

        # Initialize the tokenizer
        tokenizer = Tokenizer()

        # Fit the tokenizer on the input text
        tokenizer.fit_on_texts([inputText])

        # Convert the input text to a sequence of integers
        input_seq = tokenizer.texts_to_sequences([inputText])
        
        # Pad the input sequence to a maximum length of max_text_len
        input_seq = pad_sequences(input_seq, maxlen=max_text_len, padding='post')

        # Encode the input as state vectors.
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        
        # Generate empty target sequence of length 1
        target_seq = np.zeros((1,1))
        
        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = target_word_index['sostok']

        stop_condition = False
        decoded_sentence = ''
        
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]
            
            if(sampled_token != 'eostok'):
                decoded_sentence += ' ' + sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
                stop_condition = True

            # Update the target sequence (of length 1)
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

        return decoded_sentence
    
    ## Making the seq2seq summary
    def seq2seqsummary(input_sequence):
      newString=''
      for i in input_sequence:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
          newString=newString+reverse_target_word_index[i]+' '
      return newString

    def seq2text(input_sequence):
      newString=''
      for i in input_sequence:
        if(i!=0):
          newString=newString+reverse_source_word_index[i]+' '
      return newString

    print("Summarized:",decode_sequence(inputText))
    print("\n")


