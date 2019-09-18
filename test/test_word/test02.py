import pandas as pd
df = pd.read_csv('./data/nyt-comments/ArticlesApril2018.csv')
print(df.head())
print(df.columns)
print('열의 개수 : ', len(df.columns))

print(df['headline'].isnull().values.any()) # null값이 있는지 확인

headline = [] # 리스트 선언
headline.extend(list(df.headline.values)) # 헤드라인의 값들을 리스트로 저장
print(headline[:5]) # 상위 5개만 출력
print(type(headline))   # list

print(len(headline)) # 현재 샘플의 개수)    # 1324

headline = [n for n in headline if n != "Unknown"] # Unknown 값을 가진 샘플 제거
len(headline) # 제거 후 샘플의 개수

print(len(headline)) # 현재 샘플의 개수)    # 1214

print(headline[:5])

from string import punctuation  
def repreprocessing(s):         
    s=s.encode("utf8").decode("ascii",'ignore')
    return ''.join(c for c in s if c not in punctuation).lower() # 구두점 제거와 동시에 소문자화

text = [repreprocessing(x) for x in headline]
print(text[:5])

from keras_preprocessing.text import Tokenizer  # vocabulary를 만든다.
t = Tokenizer()
t.fit_on_texts(text)
vocab_size = len(t.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)     # 3494

sequences = list()

for line in text: # 1,214 개의 샘플에 대해서 샘플을 1개씩 가져온다.
    encoded = t.texts_to_sequences([line])[0] # 각 샘플에 대한 정수 인코딩
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print(sequences[:11]) # 11개의 샘플 출력

index_to_word={}
for key, value in t.word_index.items(): # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key

print(index_to_word[582])   # offer
print(index_to_word[82])    # big
print(index_to_word[58])    # trade
print(index_to_word[11])    # trump

max_len=max(len(l) for l in sequences)
print(max_len)  # 24                    가장 긴 샘플의 길이

# 전체 샘플의 길이를 동일하게 만드는 패딩작업 수행
from keras.preprocessing.sequence import pad_sequences
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre') 
print(sequences[:3])

# 레이블(y값 분리)
import numpy as np
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]

print(X[:3])
print(y[:3])

# 원핫인코딩
from keras.utils import to_categorical
y = to_categorical(y, num_classes=vocab_size)
print(type(y))  # np.narray
print(y.shape)  #(7803, 3494)

from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential

import os
if os.path.isfile("embedded_word01.h5") : 
    from keras.models import load_model
    model = load_model("embedded_word01.h5")
else:
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=max_len-1))    # 3494, 10, 23
    # y데이터를 분리하였으므로 이제 X데이터의 길이는 기존 데이터의 길이 - 1
    model.add(LSTM(128))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=200, verbose=2)

    model.save("embedded_word01.h5")    

model.summary()

# 문장을 생성하는 함수.
def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=23, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items(): 
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, t, 'i', 10))
print(sentence_generation(model, t, 'how', 10))


