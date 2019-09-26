# from elice_utils import EliceUtils
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from keras.models import Sequential
from keras.layers import *
# import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
pd.set_option('display.max_columns', 500)

def label_col(train, test):
    cat_columns = [col for col in train.columns if col not in ['Formatted Date', 'Temperature (C)', 'Apparent Temperature (C)','Humidity','Wind Speed (km/h)','Wind Bearing (degrees)','Pressure (millibars)','Visibility']]
    for col in tqdm(cat_columns):
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))

    column = ["Summary","Precip Type",'Temperature (C)',
            'Apparent Temperature (C)','Humidity','Wind Speed (km/h)',
            'Wind Bearing (degrees)','Pressure (millibars)']
    # train, test =np.array(train), np.array(test)
    
    # li = pd.concat([train,test],axis=0,sort=True)
    # li = pd.DataFrame(li,columns=column)
    cat0 = OneHotEncoder(categorical_features=[0])
    # cat0.fit(li)
    cat1 = OneHotEncoder(categorical_features=[1])
    train = cat0.fit_transform(train).toarray()
    test = cat0.transform(test).toarray()
    train = cat1.fit_transform(train).toarray()
    test = cat1.fit_transform(test).toarray()
    # print(train[:50])
    return np.array(train), np.array(test)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred) 
    y_true = y_true + 0.0000000001
    y_pred = y_pred + 0.0000000001
    x = np.abs((y_true - y_pred)/y_true)
    a = (100 - np.mean(x) * 100) 
    return a

def main():
    ###예측에 쓰이는 기간
    size = 100
    pre_day = 1
    # train.csv를 이용해 Visibility을 예측하는 모델을 만든후
    # test.csv의 Visibility을 예측해보세요.
    
    data_test = pd.read_csv("./_data/csv/hackerton/predict_data.csv")
    data_sample = pd.DataFrame()
    print(len(data_test["Formatted Date"]))
    data_sample['Formatted Date'] = data_test["Formatted Date"][size-1:]
    # print(data_sample)
    data_train = pd.read_csv("./_data/csv/hackerton/train_data.csv")
    train_label = data_train['Visibility']
    data_train = data_train.drop(['Formatted Date','Visibility','Precip Type'],axis=1)
    data_test = data_test.drop(['Formatted Date','Precip Type'],axis=1)
    col_train = data_train.columns
    col_test = data_test.columns
    
    
    # test.csv의 datetime 순서대로 [Formatted Date, Visibility]를 가지는 dataframe을
    # return 하도록 합니다.
   
    
    train_set, test_set = label_col(data_train,data_test)
    m = MinMaxScaler()                                      #  전처리
    train_set = m.fit_transform(train_set)
    test_set = m.transform(test_set)
 

    ##TRAIN 데이터와 LABEL데이터를 나누는 곳
    def split_(seq, size):
        aaa=[]
        for i in range(len(seq)-size+1):
            subset = seq[i:(i+size)]
            aaa.append(subset)

        print(type(aaa))
        return np.array(aaa)

    def split_label(seq, size,pre_day):
        lab = []
        lab = np.array(lab)
        if -pre_day+1 != 0:
            lab = seq[size-1:-pre_day]
        else: lab = seq[size-1:]
        print(lab.shape)
        
        for i in range(1,pre_day):
            
            # print("%d"%i,seq[size+i:-pre_day+(i+1)].shape)
            if (pre_day-1)!= i:
                lab = np.c_[lab[:], seq[size+i:-pre_day+(i+1)]]
            else:
                lab = np.c_[lab[:], seq[size+i:]]
            print(lab.shape)
        return lab
    # print(train_set.shape)

    # print(train_label[5:].shape)
   
    # test_set = np.vstack([train_set[-size:],test_set])
    train_set = split_(train_set,size)
    test_set = split_(test_set,size)
    
    
    print("test_set",test_set.shape)
    
    # train_set = train_set.reshape(train_set.shape[0],train_set.shape[1],train_set.shape[2])
    train_label = split_label(train_label,size,pre_day)
    print("train_Set",train_set.shape)
    print(train_label.shape)
    x_train, x_test, y_train, y_test = train_test_split(train_set, train_label, random_state=66,test_size=0.2)


    print(x_train.shape)#(44995, 5,6)

    model = Sequential()
    model.add(LSTM(32, input_shape=(train_set.shape[1],train_set.shape[2]),activation="relu",
                                    return_sequences=True ))
    model.add(LSTM(16,activation="relu"))
    # model.add(LSTM(64,activation="relu",return_sequences=True))
    # model.add(LSTM(64,activation="relu",return_sequences=True))
    # model.add(LSTM(100,activation="relu"))

    
    # model.add(Dense(3000, activation="relu"))
   
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    import keras
    op = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=op, loss="mae", metrics=["mae"])
    # true = pd.read_csv("./Visibility/data/sample_an.csv")
    # true = true["Visibility"]

    model.fit(x_train, y_train,batch_size=64, epochs=100)
    loss=model.evaluate(x_test,y_test)
    print(loss)

    # while 1:
    #     model.fit(x_train, y_train,batch_size=64, epochs=10, shuffle=False)
    #     loss=model.evaluate(x_test,y_test)
    #     print(loss)
    #     r2 = r2_score(y_test,model.predict(x_test))
    #     print(r2)        
    #     if r2 > 0.70:
    #         break
    #     model.reset_states()    
       

    data_sample['Visibility'] = model.predict(test_set)
    print(data_sample)
    data_sample.to_csv('./_data/csv/hackerton/predict_before.csv', index=False)
    dataframe = data_test
    predict_date = pd.read_csv("./_data/csv/hackerton/predict_date.csv")                  # read
    predict_sample = pd.merge(predict_date,data_sample,on="Formatted Date",how="left")
    print(len(predict_sample))
    predict_sample.to_csv('./_data/csv/hackerton/sample.csv', index=False)
    return dataframe

if __name__ == "__main__":
    main()


