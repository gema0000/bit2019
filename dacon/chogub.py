import pandas as pd #데이터 전처리
import numpy as np #데이터 전처리
# import pandas_profiling #데이터 전처리

import matplotlib.pyplot as plt #데이터 시각화
import seaborn as sns #데이터 시각화

sns.set_style('whitegrid')# seaborn 패키지의 배경 지정

# import missingno as msno #결측치 시각화 (없어도 현재 베이스 라인에서는 모든 작동이 가능)

from IPython.display import Image
Image('C:/Users/user/DACON Dropbox/1. 대회/11th_GIST_ETRI/7. 운영관련 분석/ipynb/baseline_basic.png')

train = pd.read_csv('./_data/dacon/train.csv')
test = pd.read_csv('./_data/dacon/test.csv')
submission = pd.read_csv('./_data/dacon/submission_1002.csv')
print('train shape: ',train.shape)
print('test shape: ',test.shape)
print('submission shape: ',submission.shape)


# 문자열인 'Time'을 datetime의 형태로 변환합니다.
# 'Time'열을 index로 전환해주는 작업을 통해, 각 세대 전력량만 활용하게 하였습니다.
train['Time'] = pd.to_datetime(train.Time) #날짜 형식으로 변환 작업을 합니다.
train = train.set_index('Time') #'Time'열을 기본 index로 설정합니다.

test['Time'] = pd.to_datetime(test.Time)
test = test.set_index('Time')

print(train.head(3))

# _, ax = plt.subplots(1,2, figsize=(15,5)) #train, test를 한 번에 비교하기 위해, 그래프 창을 2개로 만듭니다.
# # train.isnull().mean(axis=0) #각 세대별 이름과 결측치 비율이 나열됩니다.
# sns.distplot(train.isnull().mean(axis=0), ax=ax[0]) #나열된 값을 distplot을 이용해 시각화 하고, 이를 첫 번째 그래프 창에 넣습니다.
# ax[0].set_title('Distribution of Missing Values Percentage in Train set')


# sns.distplot(test.isnull().mean(axis=0), ax=ax[1]) #test data에서의 결측치 비율을 시각화 하고, 이를 두 번째 그래프 창에 넣습니다.
# ax[1].set_title('Distribution of Missing Values Percentage in Test set')
# plt.show()

answer = test.median(axis=0).sort_index() #각 세대 별 중앙에 위치한 값인 중앙값을 계산합니다.
print(answer)
print(answer.shape)     # 200,

avg_submission = submission.copy() #원본 데이터 보존을 위한 데이터 복사
avg_submission = avg_submission.set_index('meter_id') #'meter_id'를 기본 index로 설정합니다.

print(type(avg_submission)) # <class 'pandas.core.frame.DataFrame'>

for i in range(24):
    avg_submission.iloc[:,i] = answer #각 세대, 시간별 예측 값을 넣습니다. #시간별 예측 값은 중앙값으로 동일합니다
    
for i in range(10):
    avg_submission.iloc[:,24+i] = answer * 24 #각 세대, 일자별 예측값을 넣습니다.  
    
#각 세대 월 별 계산하기
months = np.zeros((answer.shape[0], 5)) # 200 * 5 로 이루어진 0 값을 사전에 만들어 놓습니다.
months[:,0] = answer * 24 * 31 # 시간별 예측값에 24를 곱해 일자별 예측값으로 만들고 31일간의 값을 구하기 위해, 31을 곱합니다.
months[:,1] = answer * 24 * 31 # 동일하게 8월을 예측하기 위해, 31을 곱합니다.
months[:,2] = answer * 24 * 30 # 동일하게 9월을 예측하기 위해, 30을 곱합니다.
months[:,3] = answer * 24 * 31 # 동일하게 10월을 예측하기 위해, 31을 곱합니다.
months[:,4] = answer * 24 * 30 # 동일하게 11월을 예측하기 위해, 30을 곱합니다.

avg_submission.iloc[:,34:] = months #미리 만들어둔 각 세대, 월별 예측값을 넣습니다.

print(avg_submission.head())

avg_submission.to_csv('./_data/dacon/dacon_baseline_1021_1.csv', index=False)
