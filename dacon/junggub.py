
from IPython.display import Image
Image('C:/Users/YHS/DACON Dropbox/1. 대회/11th_GIST_ETRI/7. 운영관련 분석/baseline_1004/baseline_image.png')

import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
import matplotlib.pyplot as plt # 데이터 시각화
import itertools
from datetime import datetime, timedelta # 시간 데이터 처리
from statsmodels.tsa.arima_model import ARIMA # ARIMA 모델
# %matplotlib inline

test = pd.read_csv("./_data/dacon/test.csv")
submission = pd.read_csv("./_data/dacon/submission_1002.csv")

test['Time'] = pd.to_datetime(test['Time']) 
test = test.set_index('Time')

print(test.head())

place_id=[]; time=[] ; target=[] # 빈 리스트를 생성합니다.
for i in test.columns:
    for j in range(len(test)):
        place_id.append(i) # place_id에 미터 ID를 정리합니다.
        time.append(test.index[j]) # time에 시간대를 정리합니다.
        target.append(test[i].iloc[j]) # target에 전력량을 정리합니다.

new_df=pd.DataFrame({'place_id':place_id,'time':time,'target':target})
new_df=new_df.dropna() # 결측치를 제거합니다.
new_df=new_df.set_index('time') # time을 인덱스로 저장합니다.
new_df.head()

# ARIMA 모델
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
print(pdq)

def get_optimal_params(y):
    
    param_dict = {}
    for param in pdq:
        try:
            model = ARIMA(y, order=param)
            results_ARIMA = model.fit(disp=-1)
            param_dict[results_ARIMA.aic] = param
        except:
            continue

    min_aic = min(param_dict.keys())
    optimal_params = param_dict[min_aic]
    return optimal_params

agg={}
for key in new_df['place_id'].unique(): # 미터ID 200개의 리스트를 unique()함수를 통해 추출합니다.
    temp = new_df.loc[new_df['place_id']==key] # 미터ID 하나를 할당합니다.
    temp_1h=temp.resample('1h').sum() # 1시간 단위로 정리합니다.
    temp_1day=temp.resample('D').sum() # 1일 단위로 정리합니다.

    # 시간별 예측
    model = ARIMA(temp_1h['target'], order=get_optimal_params(temp_1h['target'])) # AIC를 최소화하는 최적의 파라미터로 모델링합니다.
    results_ARIMA = model.fit(disp=-1)
    fcst = results_ARIMA.forecast(24) # 24시간을 예측합니다.

    a = pd.DataFrame() # a라는 데이터프레임에 예측값을 정리합니다.
    
    for i in range(24):
        a['X2018_7_1_'+str(i+1)+'h']=[fcst[0][i]] # column명을 submission 형태에 맞게 지정합니다.

        
    # 일별 예측
    model = ARIMA(temp_1day['target'], order=get_optimal_params(temp_1day['target'])) # AIC를 최소화하는 최적의 파라미터로 모델링합니다.
    results_ARIMA = model.fit(disp=-1)
    fcst = results_ARIMA.forecast(10) # 10일을 예측합니다.

    for i in range(10):
        a['X2018_7_'+str(i+1)+'_d']=[fcst[0][i]] # column명을 submission 형태에 맞게 지정합니다.
    
    
    # 월별 예측
    # 일별로 예측하여 7월 ~ 11월의 일 수에 맞게 나누어 합산합니다.
    fcst = results_ARIMA.forecast(153)
    a['X2018_7_m'] = [np.sum(fcst[0][:31])] # 7월 
    a['X2018_8_m'] = [np.sum(fcst[0][31:62])] # 8월
    a['X2018_9_m'] = [np.sum(fcst[0][62:92])] # 9월
    a['X2018_10_m'] = [np.sum(fcst[0][92:123])] # 10월
    a['X2018_11_m'] = [np.sum(fcst[0][123:153])] # 11월
    
    a['meter_id'] = key 
    agg[key] = a[submission.columns.tolist()]
    print(key)
print('---- Modeling Done ----')

output1 = pd.concat(agg, ignore_index=False)
output2 = output1.reset_index().drop(['level_0','level_1'], axis=1)
output2['id'] = output2['meter_id'].str.replace('X','').astype(int)
output2 =  output2.sort_values(by='id', ascending=True).drop(['id'], axis=1).reset_index(drop=True)
output2.to_csv('./_data/dacon/sub_baseline.csv', index=False)

