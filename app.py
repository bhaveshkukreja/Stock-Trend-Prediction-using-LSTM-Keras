import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from pandas_datareader import data 
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
from datetime import datetime

# download dataframe
x = datetime.now()


today = x.strftime("%Y-%m-%d")
start = '2010-01-01'

st.title("Stock Trend Prediction")
user_input = st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input,start,today)
#describing Data

st.subheader('Data from 2010 - Today')
st.write(df.describe())

#Visualizations
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close,'b')
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA and 200MA")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'g')
plt.plot(ma200,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)

train_data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test_data = pd.DataFrame(df['Close'][int(len(df)*0.70):])
scaler = MinMaxScaler(feature_range=(0,1))
train_data_array = scaler.fit_transform(train_data)
test_data_array = scaler.fit_transform(test_data)



#Loading my model
model=load_model('keras_model.h5')


#Test model
past_100_days = train_data.tail(100)
final_df = past_100_days.append(test_data,ignore_index =True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test , y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]

y_predicted = y_predicted *scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader("Prediction vs Original")
fig2 =plt.figure(figsize = (12,6))
plt.plot(y_test,'g', label = 'Original Price')
plt.plot(y_predicted,'r', label='Predicted Price')
plt.xlabel("Time")
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
          

