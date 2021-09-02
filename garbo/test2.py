import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import choice
#loading the model
lstm_model = load_model('new_model.h5')

dropdown = {"SCB": "Standard Chartered Bank",
                "EBL": "Everst Bank Limited",
                "NABIL": "Nabil Bank",
                "SBI": "State Bank of India",
                "NTC": "Nepal Telecom"}

print(dropdown.keys())
predicted = {}

for i in dropdown.keys():
    print(i )
    choice.predict_nepali(i)
    predicted[i] = choice.predict_nepali(i)

print(predicted)

