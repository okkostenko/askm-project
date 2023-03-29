import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_log_error
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')


TRAINING_SPLIT=0.8

def split(df):
    row_number=df.shape[0]

    global TRAIN_NUM
    TRAIN_NUM=int(row_number*TRAINING_SPLIT)

    df_train=df.iloc[:TRAIN_NUM, :] 
    df_test=df.iloc[TRAIN_NUM:, :] 

    return df_train, df_test

def losses(confirmed, df):
    actual=list(confirmed['New cases'])
    predcited=list(df['Forecast'])
    print(f'Losses: Mean Squared Log Error: {mean_squared_log_error(actual, predcited)}')
    return mean_squared_log_error(actual, predcited)

def plot_results(df_test, df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=df_test['Date'],
        y=df_test['New cases'],
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
    sarimax_model=SARIMAX(df_train['New cases'], order=(0, 1, 2), seasonal_order=(1, 2, 3, 7))
    sarimax_model_fit=sarimax_model.fit(disp=0)

    return sarimax_model_fit

def sarimax_predict(model, df_test):
    predicted=pd.DataFrame()
    forecast_test=model.forecast(len(df_test))

    predicted['Date']=df_test['Date']
    predicted['Forecast']=list(forecast_test)

    return predicted

def sarimax_test(df):

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
    df=pd.read_csv('./data/day_wise.csv')
    covid_df=df[['Date', 'New cases']].dropna()
    df_train, df_test=split(covid_df)
    model=build_model(df_train)
    return model

if __name__=='__main__':
    df=pd.read_csv('./data/day_wise.csv')
    covid_df=df[['Date', 'New cases']].dropna()
    sarimax_test(covid_df)