from flask import Flask

app = Flask(__name__)

@app.route("/<name>")
def user(name):
    return '<h1>Hello, %s !!!</h1>' %name

@app.route("/user/<name>")
def user2(name):
    return '<h1>Hello, user/%s !!!</h1>' %name  

if __name__ == '__main__' :
    app.run(host='127.0.0.1', port=5000)
    