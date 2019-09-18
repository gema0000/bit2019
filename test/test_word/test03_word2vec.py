# pip install gensim
# https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip

import re
from lxml import etree
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

targetXML=open('.//data//ted_en-20160408.xml', 'r', encoding='UTF8')
# 저자의 경우 윈도우 바탕화면에서 작업하여서 'C:\Users\USER\Desktop\ted_en-20160408.xml'이 해당 파일의 경로.  
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.

content_text = re.sub(r'\([^)]*\)', '', parse_text)
# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.

print(content_text)
print(type(content_text))     # <class 'str'>
print("요기까지 되고 요 담 부터 문제")

aaa = "i am a boy, you are a girl"

sent_text=sent_tokenize(aaa)  # sent_text의 용법을 익히자.
# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.

print("요기까지2")
print(sent_text)
print("요기까지3")
'''
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)
# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.

result=[]
result=[word_tokenize(sentence) for sentence in normalized_text]
# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.

print(result[:10])
# 문장 10개만 출력
'''

