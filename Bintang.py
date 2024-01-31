import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Judul halaman dan hiasan
st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-image: url("https://i.ibb.co/1n64TZK/background.jpg");
        background-size: cover;
        background-position: center;
        height: 100%;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """
    , unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size:50px;">Darma Alif Rakhaa</h1>
        <h2 style="font-size:30px;">Darma 👍 </h2>
    </div>
    """
    , unsafe_allow_html=True
)

# Header
st.header('Plot')
st.subheader('Sin(x) and Cos(x)')

# Generating x values from -2 to 2*pi with 1800 points
x = np.linspace(-2*np.pi, 2*np.pi, 1800)

# Calculating sin(x) and cos(x) values
y = np.sin(x)
z = np.cos(x)

# Plotting sin(x) and cos(x)
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(x, y, label='sin(x)', color='b')
ax.plot(x, z, label='cos(x)', color='g')

# Adjusting plot settings
ax.set_ylabel("")
ax.set_xlabel("x")
ax.tick_params(axis='y', labelsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.tick_params(axis='x', labelsize=15)

# Displaying the plot in Streamlit
st.pyplot(fig)

# Create two columns
col1, col2 = st.columns(2)

# Plot Sin in the first column
with col1:
    st.caption('Plot Sin')
    x_sin = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y_sin = np.sin(x_sin)

    fig_sin, ax_sin = plt.subplots(figsize=(16, 8))
    ax_sin.plot(x_sin, y_sin, label='sin(x)', color='b')
    ax_sin.set_ylabel("Sin x")
    ax_sin.set_xlabel("x")
    ax_sin.tick_params(axis='y', labelsize=20)
    ax_sin.set_xticklabels(ax_sin.get_xticklabels(), rotation=30, ha='right')
    ax_sin.tick_params(axis='x', labelsize=15)
    st.pyplot(fig_sin)

# Plot Cos in the second column
with col2:
    st.caption('Plot Cos')
    x_cos = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y_cos = np.cos(x_cos)

    fig_cos, ax_cos = plt.subplots(figsize=(16, 8))
    ax_cos.plot(x_cos, y_cos, label='cos(x)', color='g')
    ax_cos.set_ylabel("Cos x")
    ax_cos.set_xlabel("x")
    ax_cos.tick_params(axis='y', labelsize=20)
    ax_cos.set_xticklabels(ax_cos.get_xticklabels(), rotation=30, ha='right')
    ax_cos.tick_params(axis='x', labelsize=15)
    st.pyplot(fig_cos)
st.caption('Copyleft © Darma ALif Rakhaa 2024')
