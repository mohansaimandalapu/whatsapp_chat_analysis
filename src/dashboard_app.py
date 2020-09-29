import io
import streamlit as st
import pandas as pd



st.set_option('deprecation.showfileUploaderEncoding', False)

# title
GROUP_NAME = 'RandomGroup'
file_buffer = st.file_uploader("Upload WhatsApp chat text file",type='txt')
chat_text = file_buffer.readlines()

if chat_text is not None:
  st.write(chat_text)

st.title(f"{GROUP_NAME} Chat Analysis")

# chat data at a galance
st.dataframe()





st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))