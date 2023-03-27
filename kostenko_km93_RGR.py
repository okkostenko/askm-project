import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_log_error
import plotly.graph_objects as go

TRAINING_SPLIT=0.7

def split(df):
    row_number=df.shape[0]

    global TRAIN_NUM
    TRAIN_NUM=int(row_number*TRAINING_SPLIT)

    df_train=df.iloc[:TRAIN_NUM, :] 
    df_test=df.iloc[TRAIN_NUM:, :] 

    return df_train, df_test

def losses(confirmed, df):
    actual=list(confirmed['Confirmed'])
    predcited=list(df['Forecast'])
    # print(f'Losses: Mean Squared Log Error: {mean_squared_log_error(actual, predcited)}')
    # return mean_squared_log_error(actual, predcited)

def plot_results(df_test, df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=df_test['Date'],
        y=df_test['Confirmed'],
        mode='lines',
        name='Actual values'
    ))
    fig.add_trace(go.Scatter( 
        x=df.iloc[TRAIN_NUM:,0],
        y=df.iloc[TRAIN_NUM:,2],
        mode='lines',
        name='Predicted values'
    ))
    fig.show()

def build_model(df_train):
    sarimax_model=SARIMAX(df_train['Confirmed'], order=(4, 2, 0), seasonal_order=(0, 1, 1, 7))
    sarimax_model_fit=sarimax_model.fit()

    return sarimax_model_fit

def sarimax_predict(model, df_test):
    predicted=pd.DataFrame()
    forecast_test=model.forecast(len(df_test))

    predicted['Date']=df_test['Date']
    predicted['Forecast']=list(forecast_test)

    return predicted

def sarimax_test(df):
    print(f'Raw Data:\n{df.head(10)}')
    fig=px.line(df, x='Date', y='Confirmed')
    fig.show()

    df_train, df_test=split(df)

    global model
    model=build_model(df_train)
    predicted=sarimax_predict(model, df_test)

    df['Forecast']=[None]*TRAIN_NUM+list(predicted['Forecast'])
    df.plot()

    print(f'Predicted Values: \n{predicted.head(10)}')

    plot_results(df_test, df)
    plot_results(df, df)

    loss=losses(df_test, predicted)

def sarimax():
    df=pd.read_csv('day_wise.csv')
    covid_df=df[['Date', 'Confirmed']].dropna()
    df_train, df_test=split(df)
    model=build_model(df_train)
    return model

if __name__=='__main__':
    df=pd.read_csv('day_wise.csv')
    covid_df=df[['Date', 'Confirmed']].dropna()
    sarimax_test(covid_df)