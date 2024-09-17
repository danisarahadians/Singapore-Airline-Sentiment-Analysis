import streamlit as st
import pandas as pd

#import library visualisasi
import matplotlib.pyplot as plt
import seaborn as sns


def run():
    #buat judul 
    st.title('Singapore Airline Sentiment Analysis')
    

    #import gambar lewat url -> copy image address
    st.image('https://s27468.pcdn.co/wp-content/uploads/2023/06/singapore-airlines-worlds-best.jpeg', width=1000, caption='Singapore Airline')
    st.write('Dalam era digital saat ini, platform e-tiket menjadi pilihan utama bagi konsumen untuk membeli berbagai produk dan layanan. Setiap hari, ribuan ulasan atau review dari pelanggan diterima oleh platform tersebut. Ulasan-ulasan ini mengandung informasi berharga mengenai kepuasan pelanggan, kualitas produk, dan pengalaman belanja secara keseluruhan. Namun, volume ulasan yang besar membuat proses analisis manual menjadi tidak efektif dan memakan waktu. Proyek ini akan menghasilkan sebuah sistem analisis sentimen yang efektif sebesar 80% untuk ulasan pelanggan pada platform e-tiket. Dengan demikian, perusahaan dapat mengambil keputusan yang lebih tepat dan responsif terhadap kebutuhan pasar, sementara pelanggan mendapatkan manfaat dari peningkatan kualitas yang ditawarkan.')

    #nambahin subheader
    st.subheader('Wordcloud Before : ')
    st.image('bfr_wc.png')

    ############################
    # Data preprocessing untuk wordcloud
    st.subheader('Wordcloud After Pre-Processing : ')
    st.image('after_wc.png')
    ################################################################
    