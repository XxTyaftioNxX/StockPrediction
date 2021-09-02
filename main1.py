import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

#loading the dataset
df = pd.read_csv('data/SPY1.csv')

#Preparing the dataset
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

data=df.sort_index(ascending=True,axis=0)

new_dataset = data.filter(['Date','Close'], axis=1)
new_dataset.set_index('Date', inplace=True)

#Creating an array with close values
final_dataset = new_dataset.values

#Creating the Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

#Preparing traning and testing sets
train_data = final_dataset[0:987,:]
valid_data = final_dataset[987:,:]

#Scaling the data
scaled_data = scaler.fit_transform(final_dataset)

#Initializing two lists for training data
x_train_data,y_train_data = [],[]

#Creating arrays with the training array and final prediction
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

#Converting the lists into numpy array
x_train_data,y_train_data = np.array(x_train_data),np.array(y_train_data)

#Reshaping the array into 3D
x_train_data = np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

#Creating the Model
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

#Compiling and fitting the model
lstm_model.compile(loss='mean_squared_error',optimizer='adam')
res = lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

lstm_model.save("new_model.h5")
