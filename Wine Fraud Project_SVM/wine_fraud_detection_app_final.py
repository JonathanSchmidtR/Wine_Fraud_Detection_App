

import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('trained_wine_SVMclassification_model_final.sav', 'rb'))

# Set the title and favicon
st.set_page_config(page_title='Wine Fraud Detection App', page_icon=':wine_glass:')


st.title(':wine_glass: Wine Fraud Detection App :wine_glass:')
st.write("##")
st.write('This is the Wine Fraud Detection App :wine_glass: we will help you know if your wine is Legit or Fraudulent based on its chemical properties. It´s easy! Just input some info about your wine and we will do the rest :sunglasses:.')
st.write("##")

st.image('./media/wine.jpg', use_column_width=True)

st.write('---')

st.write('**Now, tell us about that wine** :clipboard:')
wine_type = st.selectbox('Wine Type', ('White Wine', 'Red Wine'))

col1, col2, col3 = st.columns(3)
with col1:
    fixed_acidity = st.number_input('Fixed Acidity', 4.6, 16.0, 7.0, 0.1)
    volatile_acidity = st.number_input('Volatile Acidity', 0.12, 1.58, 0.52, 0.01)
    citric_acid = st.number_input('Citric Acid', 0.0, 1.0, 0.25, 0.01)
    residual_sugar = st.number_input('Residual Sugar', 0.9, 15.5, 2.5, 0.1)

with col2:
    chlorides = st.number_input('Chlorides', 0.012, 0.611, 0.08, 0.001)
    free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', 1, 72, 36, 1)
    total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', 6, 289, 138, 1)
    density = st.number_input('Density', 0.990, 1.003, 0.996, 0.001)

with col3:
    pH = st.number_input('pH', 2.74, 4.01, 3.31, 0.01)
    sulphates = st.number_input('Sulphates', 0.33, 2.00, 0.68, 0.01)
    alcohol = st.number_input('Alcohol', 8.4, 14.9, 10.4, 0.1)


# Create a dictionary for the input values
#input_data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
#                  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]

# Convert the dictionary to a numpy array
#input_array = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
#                  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, wine_type]).reshape(1, -1)

if st.button("Check Authenticity"):
    input_data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
    if wine_type == "Red":
        input_data.append(1)  # Red wine
    else:
        input_data.append(0)  # White wine

    prediction = model.predict([input_data])[0]

    if prediction == 0:
        st.success("This wine is likely LEGIT! :wine_glass:")
    else:
        st.error("This wine may be FRAUDULENT! :warning:")




