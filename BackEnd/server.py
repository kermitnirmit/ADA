from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/process/<url>')
def extract_image(url):
    # show the user profile for that user
    return url.upper()
