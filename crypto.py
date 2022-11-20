#importing the libraries

import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
import math

#making the streamlit page

st.set_page_config(layout="wide")
st.write('''
# Cryto-Currency Predictor
''')

#getting the dataframe

crypto = st.selectbox('Which Crpyto-Currency would you like to be predicted?', ('BTC', 'ETH', 'DOGE', 'USDT', 'BNB', 'ADA', 'XRP', 'USDC', 'DOT1', 'HEX', 'UNI3', 'BCH', 'LTC', 'SOL1',
                                                                                'LINK', 'MATIC', 'ETC', 'THETA', 'ICP1', 'XLM', 'VET', 'FIL', 'TRX', 'XMR', 'EOS', 'SHIB', 'AAVE', 'CRO', 'BSV', 'ALGO', 'MKR', 'XTZ', 'LUNA1', 'ATOM1', 'AMP1', 'NEO', 'MIOTA', 'TFUEL', 'AVAX', 'DCR', 'CCXX',
                                                                               'HBAR', 'COMP', 'BTT1', 'KSM', 'WAVES', 'GRT2', 'TUSD', 'RUNE', 'CTC1', 'ZEC'))

current_date_time = datetime.now()
test_date_time = current_date_time - timedelta(days = 250)
period_2 = int(time.mktime(current_date_time.timetuple()))
period_1 = int(time.mktime(test_date_time.timetuple()))
interval = '1d'
url = f'https://query1.finance.yahoo.com/v7/finance/download/{crypto}-INR?period1={period_1}&period2={period_2}&interval={interval}&events=history&includeAdjustedClose=true'

df = pd.read_csv(url)
st.dataframe(df)

#                                                            PRE-PROCESSING BEGINS
#min-max scaler

df1=df.reset_index()['Close']
df1 = df1.astype(float)
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

#splitting dataset to test and train

training_size=int(len(df1)*0.60)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

#conversion of array of values into a dataset matrix

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4

time_step = 70
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()
model.add(LSTM(35,return_sequences=True,input_shape=(70,1)))
model.add(LSTM(35,return_sequences=True))
model.add(LSTM(35))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

### Lets Do the prediction and check performance metrics

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics

from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE

math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
# shift train predictions for plotting
look_back=70
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

len(test_data) 

x_input=test_data[31:].reshape(1,-1)
x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 30 days

from numpy import array

lst_output=[]
n_steps=70
i=0
while(i<30):
    
    if(len(temp_input)>70):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

day_new=np.arange(1,62)
day_pred=np.arange(62,92)

plt.plot(day_new,scaler.inverse_transform(df1[190:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
