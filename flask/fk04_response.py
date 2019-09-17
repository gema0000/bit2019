from flask import Flask
app = Flask(__name__)

from flask import make_response

@app.route('/')
def index():
    response = make_response('<h1> 잘 따라 치시오!!! </h1>')
    response.set_cookie('answer', '42')
    return response

if __name__ == "__main__" : 
    app.run(host='127.0.0.1', port=5000, debug=False)
    