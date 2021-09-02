import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd

#fucntions to predict 
import choice 
import prediction

#exception handling
pd.options.mode.chained_assignment = None  # default='warn'



#starting the server
app = dash.Dash()
server = app.server
app.title = 'Stock Prediction Analysis'

#dictionary with possible stocks
dropdown = {"SCB": "Standard Chartered Bank",
                "EBL": "Everst Bank Limited",
                "NABIL": "Nabil Bank",
                "SBI": "State Bank of India",
                "NTC": "Nepal Telecom"}

#Creating a dictionary for the predicted prices to prevent multiple loading
predicted_nepali = {}
for i in dropdown:
    predicted_nepali[i] = choice.predict_nepali(i) 

#Arranging the data from the model
actual, predicted = prediction.predict()
final = actual.filter(['Date'], axis=1)
final['Predicted'] = actual[0:987]
for i in range(987,len(actual)):
    final["Predicted"][i]=predicted[i-987]

#creating layout of the dashboard
app.layout = html.Div([   
    html.H1("Stock Price Predicton Dashboard", style={"textAlign": "center"}),   
    dcc.Tabs(id="tabs", children=[       
        dcc.Tab(label='Stock Prediction Model',children=[
			html.Div([
				html.H2("Actual Prices",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=actual.index,
								y=actual["Close"],
								mode='lines', 
                                name="Actual Price"
							),
                            go.Scatter(
                                x = final.index[987:],
                                y = final["Predicted"][987:],
                                mode = "lines",
                                name="Predicted Price"
                            )
						],
						"layout":go.Layout(
							title='',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}
				)				
			])  
        ]),

        dcc.Tab(label='Nepali Stock Prediction', children=[
            html.Div([
                html.H1("Comparision of Different Stocks", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Standard Chartered Bank', 'value': 'SCB'},
                                      {'label': 'Everst Bank Limited','value': 'EBL'}, 
                                      {'label': 'Nabil Bank', 'value': 'NABIL'}, 
                                      {'label': 'State Bank of India','value': 'SBI'},
                                      {'label': 'Nepal Telecom', 'value': 'NTC'}], 
                             multi=True,value=['SCB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
            ], className="container"),
        ])
    ])
])

#Callback for stock comparision
@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])

def update_graph(selected_dropdown):
    dropdown = {"SCB": "Standard Chartered Bank",
                    "EBL": "Everst Bank Limited",
                    "NABIL": "Nabil Bank",
                    "SBI": "State Bank of India",
                    "NTC": "Nepal Telecom"}        
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:        
        trace1.append(
          go.Scatter(x=predicted_nepali[stock].index,
                     y=predicted_nepali[stock]['Close'],
                     mode='lines', opacity=1, 
                     name=f'Actual Price of {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
         go.Scatter(x=predicted_nepali[stock].index[1800:],
                     y=predicted_nepali[stock]['Close'][1800:],
                    mode='lines', opacity=1,
                    line=dict(color="#0d0887"),
                    name=f'Predicted Price of {dropdown[stock]}',textposition='bottom center' ))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=['#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predicted Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure

if __name__=='__main__':
    app.run_server(debug=True)