import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

import warnings

warnings.filterwarnings('ignore')
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))
pdqs = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}7 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

df=pd.read_csv('day_wise.csv')
covid_df=df[['Date', 'New cases']].dropna()
row_number=df.shape[0]
TRAIN_NUM=int(row_number*0.7)
df_train=df.iloc[:TRAIN_NUM, :] 
df_test=df.iloc[TRAIN_NUM:, :] 

best_order, best_seasonal_order = sarima_optimizer_aic(df_train['New cases'], pdq, pdqs)
print(best_order, best_seasonal_order)
