


sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 
             'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train = [1,0,0,1,1,0,1]

from keras.preprocessing.text import Tokenizer
token = Tokenizer()
token.fit_on_texts(sentences)
vocab_size = len(token.word_index) +1

print(token.word_index)
print(vocab_size)

X_encoded = token.texts_to_sequences(sentences)
print(X_encoded)

max_len= max(len(l) for l in X_encoded)

from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_encoded, maxlen=max_len, padding='pre')
print(X_train)







