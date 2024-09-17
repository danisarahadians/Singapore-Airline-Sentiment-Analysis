import pandas as pd
import numpy as np
import tensorflow as tf
import re
import streamlit as st
from tensorflow.keras.models import load_model

# Library Pre-Processing
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download library
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Load the saved model
loaded_model = tf.keras.models.load_model('model_lstm_2.keras')

def run():
    resultss = {
        'Negative': 0,
        'Netral': 1,
        'Positive': 2
    }

    st.title('Menentukan Analisis Sentimen Berdasarkan Teks')

    # Menambahkan stopwords
    add_stopwords = ['to', 'I','the','a','my','and','i', 'you', 'is', 'for', 'in', 'of',
    'it', 'on', 'have', 'that', 'me', 'so', 'with', 'be', 'but',
    'at', 'was', 'just', 'I`m', 'not', 'get', 'all', 'this', 'are',
    'out', 'like', 'day', 'up', 'go', 'your', 'good', 'got', 'from',
    'do', 'going', 'no', 'now',  'work', 'will', 'about',
    'one', 'really', 'it`s', 'u', 'don`t', 'some', 'know', 'see', 'can',
    'too', 'had', 'am', 'back', '&', 'time', 'what', 'its', 'want', 'we',
    'new', 'as', 'im', 'think', 'can`t', 'if', 'when', 'an', 'more',
    'still', 'today', 'miss', 'has', 'they', 'much', 'there', 'last',
    'need', 'My', 'how', 'been', 'home', 'lol', 'off', 'Just', 'feel',
    'night', 'i`m', 'her', 'would', 'The', 'sq', 've', 'le', 'hr', 'pre', 'ca', 'th', 'b', 'uk', 'sat', 'sg', 'k', 'sfo', 'usb', 'cm', 'san', 'ft', 'hkg', 'veg', 'usd', 'rd', 'la',
    'bid', 'tag', 'usa', 'york', 'rang', 'ho', 'yr', 'fuss', 'los', 'bne', 'nap', 'hang', 'tad', 'wi', 'fi', 'sum', 'de', 'dp', 'fro', 'koh', 'samui', 'ana', 'sogi', 'prem',
    'aud', 'act', 'mnl', 'ie', 'klm', 'comb', 'ny', 'opt', 'au', 'en', 'eg', 'hub', 'yo', 'hung','sia', 'hop', 'com', 'nov', 'bc', 'cairn', 'dim', 'oct', 'bom', 'dhaka', 'cdg', 'nrt', 'cph' , 'na', 'log', 'kul', 'lh', 'w', 'ive', 'qf', 'shoe', 'tap', 'jam', 'lip', 'wan', 'oz', 'tho', 'siem', 'r', 'eve', 'melb', 'da', 'haha', 'airnz', 'coz', 'akl', 'utmost', 'gourmet', 'apps', 'mb', 'cx',  'dom', 'inr', 'pls', 'yum', 'bang', 'haagen', 'kix', 'sep', 'phee', 'rip', 'hip', 'un', 'warn', 'wee', 'z', 'ek', 'pic', 'sm', 'xmas', 'davao', 'penh', 'pcr', 'krug', 'pill', 'mar', 'ml', 'omg', 'def', 'jnb', 'kathmandu', 'pnr', 'emma', 'pudong', 'yangon', 'nang', 'qr', 'lol', 'ff', 'soo', 'so', 'vip', 'mai', 'ala', 'dxb', 'in', 'dme', 'pram', 'era', 'sim', 'bug', 'chan', 'bump', 'bent', 'pea', 'leo', 'sgn', 'amp', 'ed', 'ptv', 'dazs', 'dull', 'thr', 'aft','al', 'mad', 'pan', 'eu', 'mere', 'icing', 'danang', 'vn', 'bcn', 'singapur', 'guru', 'abit', 'fukuoka', 'wa', 'eau' , 'hoon', 'nicole', 'ham', 'ifs', 'perrier', 'sevice', 'convince', 'ref', 'easyjet', 'zrh', 'fond', 'ldn', 'ons', 'dire', 'hcmc', 'fr', 'toe', 'pond', 'ur', 'afghan', 'shenzen', 'hv', 'hkd', 'offs', 'icn', 'q', 'gaulle', 'uae', 'sooo', 'si', 'chianti', 'bengaluru', 'yeah', 'gps', 'nine', 'inc', 'jhb', 'madam', 'ban', 'signage', 'cheng', 'twg', 'alway', 'arn', 'swivel', 'krisshop', 'ya', 'ma', 'swa', 'chc', 'hyd', 'peculiar', 'oj', 'osl', 'prop', 'rhapsody', 'iam', 'wong', 'doona', 'gst', 'concoction', 'nj', 'doughy', 'fav', 'hum', 'stern', 'revamp', 'nzd', 'blunt', 'gon', 'int', 'bout', 'bento', 'hnd', 'ingham', 'bwn', 'cuz', 'jkt', 'yang', 'dr', 'mass', 'snag', 'piss', 'irate', 'adl', 'gel', 'econony', 'adjoining', 'rattle', 'chor', 'hide', 'hkt', 'amex', 'kim', 'goreng', 'singapre', 'ling', 'ap', 'damp', 'gastro', 'boss', 'temp', 'midst', 'gatwick', 'slop', 'krabi', 'sh', 'vi', 'ha', 'cmb', 'bak', 'inn', 'ful', 'ion','tbh', 'basinet', 'cab', 'andrea', 'welfare', 'kochi', 'lump', 'ashton', 'yatra', 'wotif', 'ent', 'an', 'ca', 'sang', 'ply', 'snug', 'rt', 'tongs', 'allways', 'grub', 'reckon', 'can', 'pr', 'ovo', 'maa', 'koi', 'sharifah', 'ab', 'bogus', 'nigh', 'sn', 'kat', 'david', 'john', 'savvy', 'muesli', 'ind', 'skywalk', 'imo', 'sqs', 'ng', 'teng', 'brat', 'mle', 'lye', 'iata', 'kee', 'spinal', 'hmmmm','yep', 'shin', 'gaffa' , 'chai', 'med', 'coccyx', 'eur', 'jean', 'agian', 'mee', 'kapoor', 'fog', 'sebastian', 'lingus', 'nhat', 'li', 'qi', 'saga', 'tsa', 'hagen', 'jasmine', 'ah', 'chunk', 'kebaya', 'fot', 'poc']

    # Stopwords defined
    stpwds_eng = set(stopwords.words('english'))
    stpwds_eng.update(add_stopwords)

    # Create a function for text preprocessing
    def text_preprocessing(text):
        text = text.lower()
        text = re.sub("@[A-Za-z0-9_]+", " ", text)  # Remove mentions
        text = re.sub("#[A-Za-z0-9_]+", " ", text)  # Remove hashtags
        text = re.sub(r"\\n", " ", text)            # Remove newlines
        text = re.sub(r"'s\b", "", text)            # Remove 's
        text = re.sub(r"\d+", " ", text)            # Remove numbers
        text = re.sub(r"[^\w\s]", " ", text)        # Remove punctuation
        text = text.strip()                         # Remove leading/trailing spaces
        text = re.sub(r"http\S+", " ", text)        # Remove URLs
        text = re.sub(r"\b\w{1,2}\b", " ", text)    # Remove short words
        text = re.sub("[^A-Za-z\s']", " ", text)    # Remove non-letter characters

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stpwds_eng]

        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Rejoin tokens into a single string
        return ' '.join(tokens)

    # Form input for user data
    st.write('## Input Data')
    with st.form(key='ulasan'):
        input_text = st.text_input('Review', value='')

        # Submit button
        submit = st.form_submit_button(label='Predict')

    if submit:
        # Preprocess the input
        processed_text = text_preprocessing(input_text)
        
        # Store the input and the processed text in a DataFrame
        data = pd.DataFrame({
            'riview': [input_text],
            'riview_processed': [processed_text]
        })

        # Display the original and processed text
        st.write('## Original and Processed Review')
        st.dataframe(data)

        # Ensure that the input to the model is a list of strings (1D array)
        text_input = np.array([processed_text])

        # Perform the prediction
        prediction = loaded_model.predict(text_input)

        # Get the class label from the prediction
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index

        # Display prediction result
        st.write('# Halo, berdasarkan data yang Anda input, hasil prediksi menunjukkan bahwa ulasan tersebut:')
        if predicted_class == 0:
            st.write('## Negative, Tingkatkan Pelayanan Anda sebagai Bahan Evaluasi!')
        elif predicted_class == 1:
            st.write('## Netral, Ayo Tingkatkan Pelayanan Anda agar menjadi Positive!')
        else:
            st.write('## Positive, Good job! Pertahankan kualitas pelayanan')
