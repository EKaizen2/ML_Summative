import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from smart_open import smart_open
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer

st.title("Sentiment Analysis on the Ghanaian Government")
st.header("Sentiment Analysis on the Ghanaian Government")
st.write("This web app predicts comments inputted about the Ghanaina government as to whether it is positive or negative")

#model = pickle.load(open('sentiment_model.pkl','rb'))

comment =  st.text_input("write your comment about the ghanaian govenrment recent passed budget")

st.markdown(f"my input is: {comment}")

if st.button('Predict'):
	model = pickle.load(open('sentiment_model.pkl', 'rb'))
  	x_train = openpyxl.load_workbook('test.xlsx', 'rw')
	comment_data = pd.DataFrame(x_train)
	comment_data[0] = comment
	bow_vectorizer = CountVectorizer(max_df=9000, min_df=1, max_features=513, stop_words='english')
	answer = bow_vectorizer.fit_transform(comment_data)
	prediction = model.predict(answer[0])
	if prediction.predict(answer[0]) > 0:
		print('Positive')
	elif prediction.predict(answer[0]) <= 0:
		print('Nagative')
	st.header("Please find predicted value below")
	st.write("The overall predicted score for the above player is", np.round(prediction[0]))
  	
# 	st.write("The overall predicted score for the above player is", clubs.index(club))
else:
	st.write('Thank You For Trusting Us')
