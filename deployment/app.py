import streamlit as st
import eda, predict

st.set_page_config(
    page_title='Welcome to Singapore Airline Sentiment Analysis',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.sidebar.title('Pilihan Menu: ')
select = st.sidebar.selectbox(label='Menu', options= ['EDA', 'Predict'])

if select == 'EDA':
    eda.run()
else: 
    predict.run()