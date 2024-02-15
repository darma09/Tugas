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

# Display the name of the program creator
st.header("💀Darma Alif Rakhaa💀")

pilihan = st.selectbox("Pilih operasi:", ["Penjumlahan", "Pengurangan", "Perkalian", "Pembagian", "Persentase ke Angka", "Angka ke Persentase"])

angka1 = int(st.number_input("Masukkan angka pertama:", value=1, step=1))
angka2 = int(st.number_input("Masukkan angka kedua:", value=1, step=1))

if pilihan == "Penjumlahan":
    hasil = tambah(angka1, angka2)
    st.markdown(f"**Hasil penjumlahan:** {hasil}")
elif pilihan == "Pengurangan":
    hasil = kurang(angka1, angka2)
    st.markdown(f"**Hasil pengurangan:** {hasil}")
elif pilihan == "Perkalian":
    hasil = kali(angka1, angka2)
    st.markdown(f"**Hasil perkalian:** {hasil}")
elif pilihan == "Pembagian":
    hasil = bagi(angka1, angka2)
    st.markdown(f"**Hasil pembagian:** {hasil}")
elif pilihan == "Persentase ke Angka":
    persen = float(st.number_input("Masukkan persentase:", value=0, step=0.01))
    hasil = persen / 100 * angka2
    st.markdown(f"**Hasil konversi persentase ke angka:** {hasil}")
elif pilihan == "Angka ke Persentase":
    angka = float(st.number_input("Masukkan angka:", value=0, step=1))
    persen = angka / angka2 * 100
    st.markdown(f"**Hasil konversi angka ke persentase:** {persen}%")
else:
    st.write("Pilihan tidak valid")

st.caption('Copyleft © Darma Alif Rakhaa 2024')
