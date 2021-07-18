import pandas as pd
import matplotlib.pyplot as plt

#reading the csv file into python
df = pd.read_csv('OHLC.csv')

#fix format of date
df['Date'] = df['Date'].str.replace(r'.csv$', '')

print(df.head())





