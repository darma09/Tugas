import streamlit as st
def tambah(a, b):
    return a + b

def kurang(a, b):
    return a - b

def kali(a, b):
    return a * b

def bagi(a, b):
    return a / b

st.title("KALKULATOR")

pilihan = st.selectbox("Pilih operasi:", ["Penjumlahan", "Pengurangan", "Perkalian", "Pembagian"])

angka1 = st.number_input("Masukkan angka pertama:")
angka2 = st.number_input("Masukkan angka kedua:")

if pilihan == "Penjumlahan":
    hasil = tambah(angka1, angka2)
    st.write("Hasil penjumlahan:", hasil)
elif pilihan == "Pengurangan":
    hasil = kurang(angka1, angka2)
    st.write("Hasil pengurangan:", hasil)
elif pilihan == "Perkalian":
    hasil = kali(angka1, angka2)
    st.write("Hasil perkalian:", hasil)
elif pilihan == "Pembagian":
    hasil = bagi(angka1, angka2)
    st.write("Hasil pembagian:", hasil)
else:
    st.write("Pilihan tidak valid")

st.caption('Copyleft © Darma ALif Rakhaa 2024')
