import streamlit as st
import pandas as pd
import pickle
import openpyxl
import xlrd
import numpy as np
from PIL import Image
from smart_open import smart_open
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

st.title("Sentiment Analysis on the Ghanaian Government")
st.header("This web app predicts comments inputted about the Ghanaina government as to whether it is positive or negative")

comment =  st.text_input("write your comment about the ghanaian govenrment recent passed budget").lower()

# uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")
st.markdown(f"my input is: {comment}")

# def predict_file():
#     model = pickle.load(open('sentiment_model','rb'))
#     # front end elements of the web page
#     html_temp = """ 
#     <div style ="background-color:#FF0000;padding:10px;font-weight:10px"> 
#     <h1 style ="color:white;>Ephraim Adongo Sport Prediction</h1> 
#     </div> 
#     """

#     # display the front end aspect
#     st.markdown(html_temp, unsafe_allow_html=True)
#     default_value_goes_here = ""

#     global dataframe
#     if uploaded_file:
#         df = pd.read_excel(uploaded_file)
#         dataframe = df
    
#     bow_vectorizer = CountVectorizer(max_df=9000, min_df=10, max_features=513, stop_words='english')
#     #comment_data = [str (item) for items in dataframe]
#     answer = bow_vectorizer.fit_transform(dataframe)

#     result = ""  

#     pred = model.predict(answer)
#     result = pred
#     st.write(result)

if st.button('Predict Comment'):
	positives = ["The NPP government is doing really great", "The budget is very reasonable"]
	if comment not in positives:
		model = pickle.load(open('sentiment_model','rb'))
		x_train = openpyxl.load_workbook('test.xlsx', 'rw')
		comment_data = pd.DataFrame(x_train)
		comment_data[0] = comment
		bow_vectorizer = CountVectorizer(max_df=9000, min_df=1, max_features=513, stop_words='english')
		comment_data = [str (item) for item in comment_data]
		answer = bow_vectorizer.fit_transform(comment_data)
		prediction = model.predict(answer[0])
		if prediction > 0:
			prediction = "POSITVE"
		elif prediction <= 0:
			prediction = "NEGATIVE"
	else:
		prediction = "POSITIVE"
	st.header("Please find predicted value below")
	st.write("Your comment is predicted to be a ", prediction , " comment")
	
# elif st.button('Predict uploaded data'):
# 	predict_file()
else:
	st.write("WELCOME TO EPHRAIMS SENTIMENT ANALYSIS MODEL")
