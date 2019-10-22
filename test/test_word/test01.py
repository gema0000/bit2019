# https://wikidocs.net/22660

text="""경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""

from keras_preprocessing.text import Tokenizer
token = Tokenizer()
token.fit_on_texts([text])
encoded = token.texts_to_sequences([text])
# encoded = token.texts_to_sequences([text])[0]

print(encoded)  # [[2, 3, 1, 4, 5, 6, 1, 7, 8, 1, 9, 10, 1, 11]]
print('len(encoded[0]) : {0}, {1}'.format(len(encoded[0]), "우갸갹"))

vocab_size = len(token.word_index) + 1  # 12
# 케라스 토크나이저의 정수 인코딩은 인덱스가 1부터 시작하지만,
# 케라스 원-핫 인코딩에서 배열의 인덱스가 0부터 시작하기 때문에
# 배열의 크기를 실제 단어 집합의 크기보다 +1로 생성해야하므로 미리 +1 선언 
print('단어 집합의 크기 : %d' % vocab_size)

print(token.word_index)


# 훈련 데이터를 만든다.
sequences = list()
for line in text.split('\n'): # \n을 기준으로 문장 토큰화
    encoded = token.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print('학습에 사용할 샘플의 개수: %d' % len(sequences))
print(sequences)

max_len = max(len(l) for l in sequences)    # 가장 긴 샘플의 길이가 6
print(max_len)      # 6

from keras.preprocessing.sequence import pad_sequences
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')#, value=-1) 
    # 6보다 짧은 샘플의 앞에 0으로 채운다.
    # value=-1 : 0 대신 -1로 채운다. 디폴트는 0
    # padding = 'post': 뒤쪽부터 0으로 채운다. 디폴트는 pre
    # maxlen = 3 , truncation='post' 
    # 최대길이를 3으로 지정하고 초과한 경우 각 시퀀스의 앞쪽에서 자른다. 디폴트는 pre
print(sequences)

# 샘플의 마지막 단어를 레이블로 분리.
import numpy as np
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]

print(X)
print(X.shape)  # (11, 5)
print(y)        
print(y.shape)  # (11, )

# 원핫 인코딩
from keras.utils import to_categorical
y = to_categorical(y, num_classes=vocab_size)

print(y)
print(y.shape)  # (11, 12)

## 모델링
from keras.layers import Embedding, Dense, SimpleRNN
from keras.models import Sequential

model = Sequential()    # 12, 10, 5
model.add(Embedding(vocab_size, 10, input_length=max_len-1)) # 레이블을 분리하였으므로 이제 X의 길이는 5
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)
# model.summary()

# 모델의 예측이 잘 맞고 있는지 확인하는 함수
def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=5, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items(): 
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, token, '경마장에', 4))
print(sentence_generation(model, token, '그의', 2)) # 2번 예측
print(sentence_generation(model, token, '가는', 5)) # 5번 예측

