from typing import final
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

df = pd.read_csv('data/OHLC.csv')
df['Date'] = df['Date'].str.replace(r'.csv$', '')
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']
data=df.sort_index(ascending=True,axis=0)
new_dataset = data.filter(['Date','Close', 'Symbol'], axis=1)
new_dataset.set_index('Date', inplace=True)

#making specific dataset for certain stock
new_dataset = new_dataset.loc[data['Symbol'] == "SCB"]
new_dataset = new_dataset.drop(columns='Symbol')

#loading the model
lstm_model = load_model('new_model.h5')

#Creating an array with close values
final_dataset = new_dataset.values 
    
#Creating the Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data=scaler.fit_transform(final_dataset)

####################################################################################################################

#Preparing traning and testing sets
valid_data = final_dataset[len(new_dataset)-50:,:]

#preparing for prediction
inputs_data=new_dataset[len(new_dataset)-len(valid_data) - 60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

########################################################################################################################

new_set = new_dataset[50:]

print(new_set)


print(len(predicted_closing_price))

#print(predicted_closing_price.shape)
'''final = new_dataset.filter(['Date', 'Close'], axis=1)
for i in range(len(new_dataset)-50, len(new_dataset)):
    final['Close'][i] = predicted_closing_price[i-len(new_dataset)]
'''

'''

preds = pd.DataFrame(predicted_closing_price, index=pd.date_range(start=new_dataset.index[-1]+timedelta(days=1), 
                                         periods=len(predicted_closing_price), 
                                         freq="B"), columns=[df.columns[0]])

preds = preds.reset_index()
preds = preds.set_axis(['Date', 'Close'], axis=1)
preds['Date'] = pd.to_datetime(preds.Date,format='%Y-%m-%d')
preds.set_index('Date', inplace=True)


train_data=new_dataset[:987]
valid_data=new_dataset[1866:]
valid_data['Predictions']=predicted_closing_price
plt.plot(new_dataset["Close"], label = 'Actual Price')
plt.plot(valid_data["Predictions"], label="Predicted Prices")
plt.legend()
plt.show()
'''