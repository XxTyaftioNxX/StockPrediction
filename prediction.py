import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def predict():    
    #loading the model
    lstm_model = load_model('new_model.h5')

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
    scaler.fit_transform(final_dataset)

    #Preparing traning and testing sets
    train_data = final_dataset[0:987,:]
    valid_data = final_dataset[987:,:]

    #preparing for prediction
    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.transform(inputs_data)

    X_test=[]
    for i in range(60,inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i,0])
    X_test=np.array(X_test)

    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    predicted_closing_price = lstm_model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

    predicted_closing_price = predicted_closing_price.flatten()

    return new_dataset, predicted_closing_price
    
    #used to test the predicted model
    train_data=new_dataset[:987]
    valid_data=new_dataset[987:]
    valid_data['Predictions']=predicted_closing_price
    plt.plot(new_dataset["Close"], label = 'Actual Price')
    plt.plot(valid_data["Predictions"], label="Predicted Prices")
    plt.legend()
    plt.show()



