import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Marks Predictor',page_icon='🧑🏻‍🎓👩🏻‍🎓',layout='centered')

st.title('🔖 Students Marks Predictor')
st.write('Enter The Number of Hours Studied ⌚️ (1-10) And **Click Predict** To See The Predicted Marks')

# Load The Model

def load_model(model):
   with open(model,"rb") as f:
      lin = pickle.load(f)
   return lin

try:
   model =load_model('lin.pkl')
except Exception as e:
   st.error('Your pickle file not found....')
   st.exception('Failed To Load The model...',e)
   st.stop()


hours = st.number_input('Hours_Studied',
                        min_value=1.0,
                        max_value=10.0,
                        value=4.0,
                        step=0.1,
                        format="%.1f")

if st.button('Predict'):
   try:
      X = np.array([[hours]])
      predictions = model.predict(X)
      predictions = predictions[0]

      st.success(f'Predicted Marks: {predictions:.1f}')
      st.write('Note : This Is ML Model Prediction **Result May Vary**')
   except Exception as e:
      st.error(f'Prediction Failed : {e}')
