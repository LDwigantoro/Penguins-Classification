import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Aplikasi Prediksi Pinguin 

Aplikasi web ini memprediksi spesies penguin dengan input kriteria mereka (panjang paruh, lebar paruh, panjang sirip, massa tubuh, jenis kelamin, dan pulau). Data didapat dari  [Palmer Penguins library](https://github.com/allisonhorst/palmerpenguins) dalam R oleh Allison Horst.
""")

st.sidebar.header('Fitur Input Pengguna')

st.sidebar.markdown("""
[Contoh file CSV](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Mengumpulkan fitur input pengguna ke dalam dataframe
uploaded_file = st.sidebar.file_uploader("Unggah file bertipe CSV", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Pulau',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Kelamin',('male','female'))
        bill_length_mm = st.sidebar.slider('Panjang Paruh (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Kedalaman Paruh (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Panjang Sirip (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Massa tubuh (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Menggabungkan fitur input pengguna dengan seluruh kumpulan data penguin
# berguna untuk fase pengkodean
penguins_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'], axis=1)
df = pd.concat([input_df,penguins],axis=0)

# Pengodean fitur ordinal
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Hanya memilih baris pertama (data input pengguna)

# Menampilkan Fitur Input Pengguna
st.subheader('Fitur Input Pengguna')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Menunggu file CSV untuk diunggah. Saat ini menggunakan contoh parameter input (ditunjukkan di bawah).')
    st.write(df)

# Membaca model klasifikasi yang tersimpan
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Pengaplikasian model untuk membuat prediksi
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediksi')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Probabilitas Prediksi')
st.write(prediction_proba)
