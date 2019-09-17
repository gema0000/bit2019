from flask import Flask
app = Flask(__name__)

from flask import Response, make_response
@app.route("/")
def response_test():
    custom_response = Response("Custom Response",200, {"Program": "Flask Web Application"})
    return make_response(custom_response)

@app.before_first_request
def before_first_request():
    print("앱이 기동되고 나서 첫번째 HTTP 요청에만 응답합니다.")

@app.before_request #짱
def before_request():
    print("매 HTTP 요청이 처리되기 전에 실행됩니다.")

@app.after_request
def after_request(response):
    print("매 HTTP 요청이 처리되고 나서 실행됩니다.")
    return response

@app.teardown_request   # 짱
def teardown_request(exception):
    print("매 HTTP 요청의 결과가 브라우저에응답하고 나서 호출됩니다.")

@app.teardown_appcontext
def teardown_appcontext(exception):
    print("HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행됩니다.")

if __name__ == "__main__":
    app.run(host="127.0.0.1")



