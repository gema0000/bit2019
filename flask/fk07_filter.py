from flask import Flask, render_template
app = Flask(__name__)

@app.route('/user/<name>')
def user(name):
    return render_template('user2.html', name=name)

if __name__ == "__main__" : 
    app.run(host='127.0.0.1', port=5000, debug=False)

'''
capitalize : 값의 첫 번째 문자를 대문자로 만들고 나머지는 소문자로 만든다.
lower      : 값을 소문자로 만든다.
upper      : 값을 대문자로 만든다.
title      : 값의 각 단어들을 캐피털라이즈(capitalize)한다.
trim       : 앞부분과 뒷부분에서 공백 문자를 삭제한다.
'''   