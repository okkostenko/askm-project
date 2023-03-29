import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import warnings

from model import sarimax

warnings.filterwarnings('ignore')

app = Dash(__name__)
# server=app.server()

data=pd.read_csv('./data/day_wise.csv')
covid_df=data.copy()
covid_df['Date']=pd.to_datetime(covid_df['Date']).dt.date
model=sarimax()

tab_style={
    'border':'none',
    'borderTop': '2px solid #020202',
    'backgroundColor': '#020202',
    'color': '#fff',
    'fontSize':'24px'
}
tab_style_selected={
    'backgroundColor': '#1fa0aa',
    'color': '#fff',
    'border':'none',
    'borderTop': '2px solid #1fa0aa',
    'fontSize':'24px'
}

app.layout=html.Div(children=[
    html.Div(children=[
        html.Div(children=[
            html.Img(
                    src='https://ouch-cdn2.icons8.com/ekBfgLTi0oQ13z-HmzN95Y1var5PgDX0kO-4ac9RF94/rs:fit:256:256/czM6Ly9pY29uczgu/b3VjaC1wcm9kLmFz/c2V0cy9zdmcvNDc4/LzIyOTVhMGJjLTZk/MmItNDBkMS1hNDk4/LTA5YzQ0YWU3M2Y1/OC5zdmc.png',
                    width=50,
                    height=50),
            html.H2('COVID-19 Stats')
        ],
        className='header'),
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Data Visualization', value='tab-1', style=tab_style, selected_style=tab_style_selected),
            dcc.Tab(label='Forcasting', value='tab-2', style=tab_style, selected_style=tab_style_selected)
        ]),
        html.Div(id='tab-content')
        
    ],
    className='side-bar'),
    
], className='window')

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab=='tab-1':
        return html.Div(children=[
            html.Div(children=[
            html.Label('Stats', className='title'),
            dcc.Checklist(
                ['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'],
                ['Confirmed'],
                id='checklist'
            ),

            html.Label('Date Range', className='title'),
            dcc.DatePickerRange(id='datepicker',
                                min_date_allowed=covid_df['Date'].min(),
                                max_date_allowed=covid_df['Date'].max(),
                                start_date=covid_df['Date'].min(),
                                end_date=covid_df['Date'].max())
            ],
            className='controls'),
            html.Div(children=[
                dcc.Graph(
                    id='graph',
                )
            ],
            className='main')])
    else:
        return html.Div(children=[
            html.Label('Date Range', className='title'),
            dcc.DatePickerRange(id='datepicker-forecast',
                                min_date_allowed=covid_df['Date'].min(),
                                start_date=covid_df['Date'].min(),
                                end_date=covid_df['Date'].max()),
            html.Div(children=[
                dcc.Graph(
                    id='graph-forecast',)
            ])])

@app.callback(
    Output('graph', 'figure'),
    Input('checklist', 'value'),
    Input('datepicker', 'start_date'),
    Input('datepicker', 'end_date'),
)
def update(stats, start_date, end_date):
    df=covid_df[(covid_df['Date']>=pd.to_datetime(start_date))&(covid_df['Date']<=pd.to_datetime(end_date))]
    fig=px.line(df, x='Date', y=stats, template="plotly_dark")
    return fig

@app.callback(
    Output('graph-forecast', 'figure'),
    Input('datepicker-forecast', 'start_date'),
    Input('datepicker-forecast', 'end_date'),
)
def forecast(start_date, end_date):
    dates=pd.DataFrame(pd.date_range(start=covid_df['Date'].min(), end=pd.to_datetime(end_date)), columns=['Dates'])
    dates['Dates']=dates['Dates'].dt.date

    start_idx=dates.index[dates['Dates']==pd.to_datetime(start_date)].to_list()[0]
    end_idx=dates.index[dates['Dates']==pd.to_datetime(end_date)].to_list()[0]

    dates['Forecast']=model.predict(start=start_idx, end=end_idx, dynamic= True)
    dates=dates[(dates['Dates']>=pd.to_datetime(start_date))&(dates['Dates']<=pd.to_datetime(end_date))]

    fig=px.line(dates, x='Dates', y=['Forecast'], template="plotly_dark")
    return fig

if __name__=='__main__':
    app.run_server(debug=False)
