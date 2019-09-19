from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# token = Tokenizer(num_words=1000)
token = Tokenizer()
token.fit_on_texts(samples)
print(token.word_index)

vocab_size = len(token.word_index) + 1
print(vocab_size)   # 10

sequences = token.texts_to_sequences(samples)
print(sequences)

one_hot_results = token.texts_to_matrix(samples, mode='binary')

word_index = token.word_index
print('%s개의 고유한 토큰을 찾았습니다.' % len(word_index)) # 9

print(one_hot_results)
print(one_hot_results.shape)
print(type(one_hot_results))

