import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set the title of the app
st.title("Laptop Predictor")

# Create a sidebar to group the input fields together
with st.sidebar:
    # Brand
    company = st.selectbox('Brand', df['Company'].unique())
    # Type of laptop
    type = st.selectbox('Type', df['TypeName'].unique())
    # RAM
    ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    # Weight
    weight = st.number_input('Weight of the Laptop')
    # Touchscreen
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    # IPS
    ips = st.selectbox('IPS', ['No', 'Yes'])
    # Screen size
    screen_size = st.number_input('Screen Size')
    # Screen resolution
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    # CPU
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())
    # HDD
    hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
    # SSD
    ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
    # GPU
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())
    # OS
    os = st.selectbox('OS', df['os'].unique())

# Store the user's input values in the session state
st.session_state.company = company
st.session_state.type = type
st.session_state.ram = ram
st.session_state.weight = weight
st.session_state.touchscreen = 1 if touchscreen == 'Yes' else 0  # Convert to numerical
st.session_state.ips = 1 if ips == 'Yes' else 0  # Convert to numerical
st.session_state.screen_size = screen_size
st.session_state.resolution = resolution
st.session_state.cpu = cpu
st.session_state.hdd = hdd
st.session_state.ssd = ssd
st.session_state.gpu = gpu
st.session_state.os = os

# Calculate the PPI
ppi = ((int(resolution.split('x')[0])**2) + (int(resolution.split('x')[1])**2))**0.5 / screen_size

# Make the prediction
query = np.array([st.session_state.company, st.session_state.type, st.session_state.ram, st.session_state.weight, st.session_state.touchscreen, st.session_state.ips, ppi, st.session_state.cpu, st.session_state.hdd, st.session_state.ssd, st.session_state.gpu, st.session_state.os])
query = query.reshape(1, 12)

# Show a progress bar while the prediction is being processed
with st.spinner("Predicting price..."):
    prediction = pipe.predict(query)[0]

# Convert the prediction to an integer
prediction = int(prediction)

# Display the prediction
st.title("The predicted price of this configuration is " + str(prediction))
