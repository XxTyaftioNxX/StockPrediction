import choice 
import prediction

dropdown = {"SCB": "Standard Chartered Bank",
                "EBL": "Everst Bank Limited",
                "NABIL": "Nabil Bank",
                "SBI": "State Bank of India",
                "NTC": "Nepal Telecom"}

predicted = {}
actual, pre = prediction.predict()

print(actual.index)
for i in dropdown:
    predicted[i] = choice.predict_nepali(i) 

print(predicted['NABIL'].index)
print(predicted['NABIL'].values)