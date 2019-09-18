# https://wikidocs.net/33793
# from keras.layers import Embedding
# v = Embedding(20001, 256, input_length=500) # 전체 단어 집합의 크기, 아웃풋, 인풋컬럼

sentences = ['멋있어 최고야 짱이야 감탄이다 우갸갹', '헛소리 짱이야 지껄이네', '닥쳐 자식아', '우와 대단하다', 
             '우수한 성적', '성적 형편없다', '최상의 멋있어 퀄리티 므흣']
y_train = [1,0,0,1,1,0,1]

from keras.preprocessing.text import Tokenizer
token = Tokenizer()
token.fit_on_texts(sentences)

print(token)
print(type(token))

print(token.word_index)
print(token.word_counts)

vocab_size = len(token.word_index) + 1
print(vocab_size)   # 18

X_encoded = token.texts_to_sequences(sentences)
print(X_encoded)

max_len = max(len(l) for l in X_encoded)
print(max_len)  # 5


from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_encoded, maxlen=max_len, padding='pre')
print(X_train)
print(X_train.shape)    # (7, 5)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(vocab_size, 77, input_length=max_len)) # (20,77,5)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)




